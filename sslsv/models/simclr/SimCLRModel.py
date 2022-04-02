from torch import nn
from sslsv.models.baseModel.BaseModel import BaseModel

class SimCLRModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        self.mlp = nn.Sequential(
            nn.Linear(1024, self.mlp_dim),
            nn.BatchNorm1d(self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.BatchNorm1d(self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim)
        )