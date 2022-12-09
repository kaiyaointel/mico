# IMPORT

import numpy as np
import torch
import csv
import os
from tqdm.notebook import tqdm
from mico_competition import ChallengeDataset, load_cifar10, load_model


# ATTACK

CHALLENGE = "cifar10_kit"
LEN_TRAINING = 50000
LEN_CHALLENGE = 100

scenarios = os.listdir(CHALLENGE)
phases = ['dev', 'final', 'train']

dataset = load_cifar10(dataset_dir="./cifar10_data")

criterion = torch.nn.CrossEntropyLoss(reduction='none')

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from collections import OrderedDict
from typing import List, Optional, Union, Type, TypeVar
from mico_competition import ChallengeDataset, load_cifar10, load_model
from torch.utils.data import DataLoader

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

attack_model = DNN()
attack_model.load_state_dict(torch.load('attack_model.pt'))
attack_model.eval()

for scenario in tqdm(scenarios, desc="scenario"):
    for phase in tqdm(phases, desc="phase"):
        root = os.path.join(CHALLENGE, scenario, phase)
        for model_folder in tqdm(sorted(os.listdir(root), key=lambda d: int(d.split('_')[1])), desc="model"):
            path = os.path.join(root, model_folder)
            challenge_dataset = ChallengeDataset.from_path(path, dataset=dataset, len_training=LEN_TRAINING)
            challenge_points = challenge_dataset.get_challenges()

            # This is where you plug in your membership inference attack
            # As an example, here is a simple loss threshold attack

            # Loss Threshold Attack
            model = load_model('cifar10', path)
            challenge_dataloader = torch.utils.data.DataLoader(challenge_points, batch_size=2*LEN_CHALLENGE)
            features, labels = next(iter(challenge_dataloader))
            
            output = model(features)
            # print("output shape: ", output.shape) # torch.Size([200, 10])
            loss = criterion(output, labels)
            # print("loss shape: ", loss.shape) # torch.Size([200])
            
            f = torch.cat((output, loss.unsqueeze(1)), 1)
            with torch.no_grad():
                predictions = attack_model(f)

            # predictions = -loss.detach().numpy() # this predictions is membership predictions
            # predictions = torch.var(torch.abs(output), 1).detach().numpy() # using the std of output to indicate possibility of being a member
            # print("predictions: ", predictions.shape) # (200,)
            # Normalize to unit interval

            predictions = predictions.squeeze()

            predictions = predictions.detach().numpy()

            # min_prediction = np.min(predictions)
            # max_prediction = np.max(predictions)
            # predictions = (predictions - min_prediction) / (max_prediction - min_prediction)

            assert np.all((0 <= predictions) & (predictions <= 1))

            with open(os.path.join(path, "prediction.csv"), "w") as f:
                 csv.writer(f).writerow(predictions)


# SCORING

from mico_competition.scoring import tpr_at_fpr, score, generate_roc, generate_table
from sklearn.metrics import roc_curve, roc_auc_score

FPR_THRESHOLD = 0.1

all_scores = {}
phases = ['train']

for scenario in tqdm(scenarios, desc="scenario"): 
    all_scores[scenario] = {}    
    for phase in tqdm(phases, desc="phase"):
        predictions = []
        solutions  = []

        root = os.path.join(CHALLENGE, scenario, phase)
        for model_folder in tqdm(sorted(os.listdir(root), key=lambda d: int(d.split('_')[1])), desc="model"):
            path = os.path.join(root, model_folder)
            predictions.append(np.loadtxt(os.path.join(path, "prediction.csv"), delimiter=","))
            solutions.append(np.loadtxt(os.path.join(path, "solution.csv"),   delimiter=","))

        predictions = np.concatenate(predictions)
        solutions = np.concatenate(solutions)
        
        scores = score(solutions, predictions)
        all_scores[scenario][phase] = scores

# plot TPR/FPR
import matplotlib.pyplot as plt
import matplotlib

for scenario in scenarios:
    fpr = all_scores[scenario]['train']['fpr']
    tpr = all_scores[scenario]['train']['tpr']
    fig = generate_roc(fpr, tpr)
    fig.suptitle(f"{scenario}", x=-0.1, y=0.5)
    fig.tight_layout(pad=1.0)

# score
import pandas as pd
from IPython.display import display

for scenario in scenarios:
    print(scenario)
    scores = all_scores[scenario]['train']
    scores.pop('fpr', None)
    scores.pop('tpr', None)
    display(pd.DataFrame([scores]))


# SUBMISSION

import zipfile

phases = ['dev', 'final']

with zipfile.ZipFile("predictions_cifar10.zip", 'w') as zipf:
    for scenario in tqdm(scenarios, desc="scenario"): 
        for phase in tqdm(phases, desc="phase"):
            root = os.path.join(CHALLENGE, scenario, phase)
            for model_folder in tqdm(sorted(os.listdir(root), key=lambda d: int(d.split('_')[1])), desc="model"):
                path = os.path.join(root, model_folder)
                file = os.path.join(path, "prediction.csv")
                if os.path.exists(file):
                    zipf.write(file)
                else:
                    raise FileNotFoundError(f"`prediction.csv` not found in {path}. You need to provide predictions for all challenges")
