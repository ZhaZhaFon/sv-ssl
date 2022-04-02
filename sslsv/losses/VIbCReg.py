import torch
from torch import nn
import torch.nn.functional as F

class VIbCReg(nn.Module):
    def __init__(self, inv_weight=1.0, var_weight=1.0, cov_weight=0.04):
        super().__init__()
        self.inv_weight = inv_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight

    def forward(self, data):
        Z_a, Z_b = data

        N, D = Z_a.size()

        # Invariance loss
        inv_loss = F.mse_loss(Z_a, Z_b)

        # Variance loss
        Z_a_std = torch.sqrt(Z_a.var(dim=0) + 1e-04)
        Z_b_std = torch.sqrt(Z_b.var(dim=0) + 1e-04)
        var_loss = torch.mean(F.relu(1 - Z_a_std))
        var_loss += torch.mean(F.relu(1 - Z_b_std))

        # Covariance loss
        Z_a = Z_a - Z_a.mean(dim=0)
        Z_b = Z_b - Z_b.mean(dim=0)
        Z_a_norm = F.normalize(Z_a, p=2, dim=0)# Z_a / torch.linalg.matrix_norm(Z_a) #the norm along the batch dimension ?
        Z_b_norm = F.normalize(Z_b, p=2, dim=0)
        Z_a_cov = torch.mm(Z_a_norm.T, Z_a_norm)
        Z_b_cov = torch.mm(Z_b_norm.T, Z_b_norm)

        # diag = torch.eye(D, dtype=torch.bool, device=Z_a.device)
        Z_a_cov.fill_diagonal_(0.0)
        Z_b_cov.fill_diagonal_(0.0)
        # cov_loss = Z_a_cov[~diag].pow_(2).sum() / (D ** 2)
        # cov_loss += Z_b_cov[~diag].pow_(2).sum() / (D ** 2)
        cov_loss = (Z_a_cov ** 2).mean()
        cov_loss += (Z_b_cov ** 2).mean()

        loss = self.inv_weight * inv_loss
        loss += self.var_weight * var_loss
        loss += self.cov_weight * cov_loss

        return loss