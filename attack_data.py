from mico_competition import ChallengeDataset, load_cifar10, load_model
import numpy as np
from torch.utils.data import DataLoader

full_dataset = load_cifar10(dataset_dir=".", download=False)

with open("seed_training", "r") as f:
    seed_training = int(f.read())
    print("seed_training: ", seed_training)
with open("seed_challenge", "r") as f:
    seed_challenge = int(f.read())
    print("seed_challenge: ", seed_challenge)
with open("seed_membership", "r") as f:
    seed_membership = int(f.read())
    print("seed_membership: ", seed_membership)

challenge_dataset = ChallengeDataset(
    full_dataset,
    len_challenge=100,
    len_training=50000,
    seed_challenge=seed_challenge,
    seed_training=seed_training,
    seed_membership=seed_membership)

train_dataset = challenge_dataset.get_train_dataset()
eval_dataset = challenge_dataset.get_eval_dataset()

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=1,
)

model = load_model('cifar10', ".")

model.eval()

device = "cpu"

import torch
from torch import nn

feature = []
membership = []

# member
with torch.no_grad():
    count = 0
    count_correct = 0
    for i, (inputs, target) in enumerate(train_loader):
        inputs = inputs.to(device)
        target = target.to(device)

        output = model(inputs)
        
        feature.append(output)
        membership.append(torch.tensor([1]))
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)

        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()

        count += 1
        if preds == labels:
            count_correct += 1

        if count == 100:
            print(count_correct / 100)
            break

# non-member
with torch.no_grad():
    count = 0
    count_correct = 0
    for i, (inputs, target) in enumerate(eval_loader):
        inputs = inputs.to(device)
        target = target.to(device)

        output = model(inputs)
        
        feature.append(output)
        membership.append(torch.tensor([0]))

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)

        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()

        count += 1
        if preds == labels:
            count_correct += 1

        if count == 100:
            print(count_correct / 100)
            break

# attack

from torch.utils.data import Dataset

class AttackDataset(Dataset):
    def __init__(self, feature, membership):
        super().__init__()
        self.feature = feature
        self.membership = membership

    def __len__(self):
        return len(self.membership)

    def __getitem__(self, index):
        this_feature = self.feature[index]
        this_membership = self.membership[index]
        return this_feature, this_membership


attack_dataset = AttackDataset(feature, membership)

attack_train_loader = DataLoader(
    attack_dataset,
    batch_size=10,
    shuffle=True,
)

import os
import torch
import torch.nn as nn

from collections import OrderedDict
from typing import List, Optional, Union, Type, TypeVar

X = TypeVar("X", bound="DNN")

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.Linear(in_features=10, out_features=5),
            nn.ReLU(),
            nn.Linear(in_features=5, out_features=3),
            nn.ReLU(),
            nn.Linear(in_features=3, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.dnn(x)
        return logits

model = DNN()

import torch.optim as optim
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0)

model.train()

criterion = nn.BCELoss()

# train attack
for i in range(10000):
    losses = []
    for i, (inputs, target) in enumerate(attack_train_loader):
        inputs = inputs.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model(inputs)

        loss = criterion(output, target.unsqueeze(1).float()) # notice here need to unsqueeze and float to deal with dimension
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()

    print(np.mean(losses))
    
# evaluate attack
for i, (inputs, target) in enumerate(attack_train_loader):
    output = model(inputs)
    print(output, target)
