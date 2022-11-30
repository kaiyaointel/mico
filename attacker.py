from __future__ import annotations

import os
import torch
import torch.nn as nn

from collections import OrderedDict
from typing import List, Optional, Union, Type, TypeVar
from torch.utils.data import Dataset, ConcatDataset, random_split

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Linear(in_features=10, out_features=5)
            nn.Linear(in_features=5, out_features=3)
            nn.Linear(in_features=3, out_features=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape of x is [10]
        logits = self.dnn(x)
        return logits

    @classmethod
    def load(cls: Type[X], path: Union[str, os.PathLike]) -> X:
        model = cls()
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict((k.replace('_module.', ''), v) for k, v in state_dict.items())
        model.load_state_dict(new_state_dict)
        model.eval()
        return model
