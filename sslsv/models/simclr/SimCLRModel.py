import torch
from torch import nn
import torch.nn.functional as F

from sslsv.encoders.ThinResNet34 import ThinResNet34

from sslsv.losses.InfoNCE import InfoNCE
from sslsv.losses.VICReg import VICReg
from sslsv.losses.BarlowTwins import BarlowTwins

from sslsv.models.baseModel import BaseModel


class SimCLRModel(BaseModel.Model):

    def __init__(self, config):
        super().__init__(config)

        self.encoder = ThinResNet34()
        self.mlp = nn.Sequential(
            nn.Linear(1024, self.mlp_dim),
            nn.BatchNorm1d(self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.BatchNorm1d(self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim)
        )