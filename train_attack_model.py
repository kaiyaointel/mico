import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from collections import OrderedDict
from typing import List, Optional, Union, Type, TypeVar
from mico_competition import ChallengeDataset, load_cifar10, load_model
from torch.utils.data import DataLoader


feature = torch.load('feature')
membership = torch.load('membership')

device = "cpu"

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
attack_train_set, attack_eval_set = torch.utils.data.random_split(attack_dataset, [1000, 1000])

attack_train_loader = DataLoader(
    attack_train_set,
    batch_size=10,
    shuffle=True,
)
attack_eval_loader = DataLoader(
    attack_eval_set,
    batch_size=10,
    shuffle=True,
)

X = TypeVar("X", bound="DNN")

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.Linear(in_features=10, out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.dnn(x)
        return logits

model = DNN()

optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1)

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

    print(np.mean(losses)) # mean loss of one epoch

# evaluate attack on train
print("evaluate attack on train ...")
model.eval()
for i, (inputs, target) in enumerate(attack_train_loader):
    output = model(inputs)
    print(output, target)

# evaluate attack on eval
print("evaluate attack on eval ...")
model.eval()
for i, (inputs, target) in enumerate(attack_eval_loader):
    output = model(inputs)
    print(output, target)
