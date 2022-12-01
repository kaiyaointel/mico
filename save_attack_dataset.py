import torch
import numpy as np
from torch import nn
from mico_competition import ChallengeDataset, load_cifar10, load_model
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
print(train_dataset.__len__())
eval_dataset = challenge_dataset.get_eval_dataset()
print(eval_dataset.__len__())

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

features = []
memberships = []

data_size = 1000

# member
with torch.no_grad():
    count = 0
    count_correct = 0
    for i, (inputs, target) in enumerate(train_loader):
        print(i)
        inputs = inputs.to(device)
        target = target.to(device)

        output = model(inputs)
        
        # features.append(output)
        memberships.append(torch.tensor([1])) # member: 1
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        
        feature = torch.cat((output.squeeze(), loss.view(1))) # output + loss
        features.append(feature)

        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()

        count += 1
        if preds == labels:
            count_correct += 1

        if count == data_size:
            print(count_correct / data_size)
            break

# non-member
with torch.no_grad():
    count = 0
    count_correct = 0
    for i, (inputs, target) in enumerate(eval_loader):
        print(i)
        inputs = inputs.to(device)
        target = target.to(device)

        output = model(inputs)
        
        # features.append(output)
        memberships.append(torch.tensor([0])) # non-member: 0

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        
        feature = torch.cat((output.squeeze(), loss.view(1))) # output + loss
        features.append(feature)

        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()

        count += 1
        if preds == labels:
            count_correct += 1

        if count == data_size:
            print(count_correct / data_size)
            break

torch.save(features, 'features')
torch.save(memberships, 'memberships')
