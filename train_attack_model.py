import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from collections import OrderedDict
from typing import List, Optional, Union, Type, TypeVar
from mico_competition import ChallengeDataset, load_cifar10, load_model
from torch.utils.data import DataLoader

torch.manual_seed(2022)

features = torch.load('features')
memberships = torch.load('memberships')

device = "cpu"

# attack

from torch.utils.data import Dataset

class AttackDataset(Dataset):
    def __init__(self, features, memberships):
        super().__init__()
        self.features = features
        self.memberships = memberships

    def __len__(self):
        return len(self.memberships)

    def __getitem__(self, index):
        this_feature = self.features[index]
        this_membership = self.memberships[index]
        return this_feature, this_membership


attack_dataset = AttackDataset(features, memberships)
attack_train_set, attack_eval_set = torch.utils.data.random_split(attack_dataset, [1800, 200])

batch_size = 50

attack_train_loader = DataLoader(
    attack_train_set,
    batch_size=batch_size,
    shuffle=True,
)
attack_eval_loader = DataLoader(
    attack_eval_set,
    batch_size=batch_size,
    shuffle=True,
)

X = TypeVar("X", bound="DNN")

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.Linear(11, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.dnn(x)
        return logits

model = DNN()

optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

criterion = nn.BCELoss()

# train attack
for i in range(10000):

    model.train()
    losses = []
    correct_sample = 0
    total_sample = 0
    for i, (inputs, target) in enumerate(attack_train_loader):
        inputs = inputs.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model(inputs)
        
        loss = criterion(output, target.float()) # notice here need to unsqueeze and float to deal with dimension
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()

        output = torch.where(output >= 0.5, torch.ones_like(output), output)
        output = torch.where(output < 0.5, torch.zeros_like(output), output)

        for k in range(batch_size):
            total_sample += 1
            if output.squeeze()[k] == target.squeeze()[k]:
                correct_sample += 1

    model.eval()
    with torch.no_grad():
        losses_eval = []
        correct_sample_eval = 0
        total_sample_eval = 0
        for i, (inputs, target) in enumerate(attack_eval_loader):
            output = model(inputs)

            loss_eval = criterion(output, target.float())
            losses_eval.append(loss_eval.item())

            output = torch.where(output >= 0.5, torch.ones_like(output), output)
            output = torch.where(output < 0.5, torch.zeros_like(output), output)

            for k in range(batch_size):
                total_sample_eval += 1
                if output.squeeze()[k] == target.squeeze()[k]:
                    correct_sample_eval += 1

    print("train_loss ", np.mean(losses), " train_acc ", correct_sample / total_sample,
          " eval_loss ", np.mean(losses_eval), " eval_acc ", correct_sample_eval / total_sample_eval,) # mean loss of one epoch

torch.save(model, 'attack_model.pt')

# evaluation
with torch.no_grad():
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
