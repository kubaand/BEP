import logging
import timm
import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union, List, Final
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import named_apply, checkpoint_seq, adapt_input_conv
from timm.models._registry import generate_default_cfgs, register_model, register_model_deprecations

__all__ = ['VisionTransformer']  # model_registry will add each entrypoint fn to this


_logger = logging.getLogger(__name__)

from timm.models.vision_transformer import VisionTransformer, _cfg
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import random
import numpy as np
from torch.utils.data import Subset, DataLoader
from datasets import load_dataset


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, data):
        if len(self.buffer) + len(data) > self.buffer_size:
            self.buffer = self.buffer[len(data):]
        self.buffer.extend(data)

    def get_data(self):
        return random.sample(self.buffer, min(len(self.buffer), self.buffer_size))



buffer_size = 1000
replay_buffer = ReplayBuffer(buffer_size)


def custom_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels

def update_replay_buffer(replay_buffer, dataset, buffer_update_size=100):
    indices = np.random.choice(len(dataset), buffer_update_size, replace=False)
    replay_data = [dataset[i] for i in indices]
    replay_buffer.add(replay_data)

def get_mixed_dataloader(task_dataset, replay_buffer, batch_size=32):

    replay_data = replay_buffer.get_data()
    replay_images, replay_labels = zip(*replay_data)

    if isinstance(replay_images, tuple):
        replay_images = torch.stack(replay_images)   
    replay_labels = torch.tensor(replay_labels)
    replay_dataset = torch.utils.data.TensorDataset(replay_images, replay_labels)

    mixed_dataset = torch.utils.data.ConcatDataset([task_dataset, replay_dataset])
    return torch.utils.data.DataLoader(mixed_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

def get_dataset_splits(dataset_name, root='./data10', transform=None, num_tasks=20, classes_per_task=5):
    if dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    targets = np.array(dataset.targets)
    tasks_train, tasks_test = {}, {}
    for task_id in range(num_tasks):
        class_indices = range(task_id * classes_per_task, (task_id + 1) * classes_per_task)
        task_indices = np.where(np.isin(targets, class_indices))[0]
        tasks_train[task_id] = Subset(dataset, task_indices)

        test_targets = np.array(test_dataset.targets)
        test_task_indices = np.where(np.isin(test_targets, class_indices))[0]
        tasks_test[task_id] = Subset(test_dataset, test_task_indices)
    return tasks_train, tasks_test

def train_model(model, train_loader, optimizer, criterion, num_epochs=7):
    model.train()
    for epoch in range(num_epochs):
        running_loss=0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def update_task_performance(task_accuracies, best_task_accuracies, current_task, current_accuracy):
    if current_task not in task_accuracies:
        task_accuracies[current_task] = []
    task_accuracies[current_task].append(current_accuracy)
    if current_task not in best_task_accuracies or current_accuracy > best_task_accuracies[current_task]:
        best_task_accuracies[current_task] = current_accuracy

def calculate_forgetting(task_accuracies, best_task_accuracies):
    forgettings = []
    for task, accuracies in task_accuracies.items():
        best_accuracy = best_task_accuracies[task]
        final_accuracy = accuracies[-1]
        forgetting = max(0, best_accuracy - final_accuracy)
        forgettings.append(forgetting)
    return np.mean(forgettings) if forgettings else 0


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


dataset_name = 'cifar100'
tasks_train, tasks_test = get_dataset_splits(dataset_name, transform=transform)

class CustomVisionTransformer(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, *args, window_size = 8, stride = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.window_size = 3
        self.stride = 1

        self.conv_sliding_window = nn.Conv2d(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim,
            kernel_size=self.window_size,
            stride=self.stride,
            padding=1,
            groups=self.embed_dim
        )
    def apply_sliding_window(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv_sliding_window(x)

        return x
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        B, N, C = x.shape
        H, W = int(math.sqrt(N)), int(math.sqrt(N))
        x = x.transpose(1, 2).view(B, C, H, W)
        

        x = self.apply_sliding_window(x)

        x = x.view(B, N, C)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x


model = CustomVisionTransformer(
    img_size=224,
    patch_size=16,
    embed_dim=192,
    depth=12,
    num_heads=3,
    window_size=8,
    stride=4)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0)
criterion = torch.nn.CrossEntropyLoss()


task_accuracies = {}
best_task_accuracies = {}


for task_name, dataset in tasks_train.items():
    update_replay_buffer(replay_buffer, dataset, buffer_update_size=400)
    mixed_train_loader = get_mixed_dataloader(dataset, replay_buffer, batch_size=32)
    train_model(model, mixed_train_loader, optimizer, criterion)

    for seen_task_name, seen_dataset in tasks_test.items():
        if seen_task_name <= task_name:
            test_loader = DataLoader(seen_dataset, batch_size=100, shuffle=False)
            accuracy = evaluate_model(model, test_loader)
            update_task_performance(task_accuracies, best_task_accuracies, seen_task_name, accuracy)


average_accuracy = np.mean([task_accuracies[task][-1] for task in task_accuracies])
learning_accuracy = task_accuracies[list(tasks_train.keys())[-1]][-1]
joint_accuracy = evaluate_model(model, DataLoader(torch.utils.data.ConcatDataset(list(tasks_test.values())), batch_size=100, shuffle=False))
average_forgetting = calculate_forgetting(task_accuracies, best_task_accuracies)

print(f"Average Accuracy: {average_accuracy}")
print(f"Learning Accuracy: {learning_accuracy}")
print(f"Joint/Multi-task Accuracy: {joint_accuracy}")
print(f"Average Forgetting: {average_forgetting}")