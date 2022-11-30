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

with torch.no_grad():
    count = 0
    count_correct = 0
    for i, (inputs, target) in enumerate(train_loader):
        inputs = inputs.to(device)
        target = target.to(device)

        output = model(inputs)
        
        feature.append(output)
        membership.append(1)
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)

        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()

        count += 1
        if preds == labels:
            count_correct += 1

        if count == 10:
            print(count_correct / 10)
            break

with torch.no_grad():
    count = 0
    count_correct = 0
    for i, (inputs, target) in enumerate(eval_loader):
        inputs = inputs.to(device)
        target = target.to(device)

        output = model(inputs)
        
        feature.append(output)
        membership.append(0)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)

        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()

        count += 1
        if preds == labels:
            count_correct += 1

        if count == 10:
            print(count_correct / 10)
            break

from torch.utils.data import Dataset

class AttackDataset(Dataset):
    def __init__(self, feature, membership):
        super().__init__()
        self.feature = feature
        self.membership = membership

    def __len__(self):
        return len(self.membership)

    def __getitem__(self, index):
        this_feature = feature[i]
        this_membership = membership[i]
        return this_feature, this_membership

attack_dataset = AttackDataset(feature, membership)

attack_train_loader = DataLoader(
    attack_dataset,
    batch_size=1,
)

for i, (inputs, target) in enumerate(attack_train_loader):
    print(inputs)
    print(target)
