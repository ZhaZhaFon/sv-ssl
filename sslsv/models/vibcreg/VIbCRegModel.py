from torch import nn

from sslsv.models.baseModel.BaseModel import BaseModel
from sslsv.losses.VIbCReg import VIbCReg
from sslsv.layers.IterNorm import IterNorm

class VIbCRegModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        self.vibcreg = VIbCReg(
            config.vic_inv_weight,
            config.vic_var_weight,
            config.vic_cov_weight            
        )
        self.mlp = nn.Sequential(
            nn.Linear(1024, self.mlp_dim),
            nn.BatchNorm1d(self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.BatchNorm1d(self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            IterNorm(self.mlp_dim, affine=False)
        )

    def compute_loss_(self, Z_1, Z_2, _):
        loss = self.vibcreg((Z_1, Z_2))
        _, accuracy = self.infonce((Z_1, Z_2))
        return loss, accuracy