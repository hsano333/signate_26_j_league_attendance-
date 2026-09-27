import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import IntEnum


D_in = 120
H = 4
D_out = 1
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2


class Diff(IntEnum):
    NEG = 0
    ZERO = 1
    POS = 2


class Model(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.5) -> None:
        super(Model, self).__init__()
        self.D_in = input_size
        self.D_out = output_size
        self.dropout1 = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

        self.linear00 = torch.nn.Linear(self.D_in, int(self.D_in / 2))
        self.linear01 = torch.nn.Linear(int(self.D_in / 2), D_out)
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        # print(f"{x.shape=}, {x=}")
        # print(x.min().item(), x.max().item(), x.mean().item(), x.std().item())
        x = self.relu(self.linear00(x))
        x = self.linear01(x)
        return x
