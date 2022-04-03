from sslsv.models.simclr.SimCLRModel import SimCLRModel

from torch.cuda.amp import autocast

import functools, operator, collections

class SimCLRModelMC_2x2(SimCLRModel):
    def __init__(self, config):
        super().__init__(config)

    def compute_loss_MC(self, Z_1, Z_2, Z_3, Z_4):
        metrics_list = [[]] * 4

        loss1, metrics_list[0] = self.compute_loss(Z_1, Z_3)
        loss2, metrics_list[1] = self.compute_loss(Z_1, Z_4)
        loss3, metrics_list[2] = self.compute_loss(Z_2, Z_3) 
        loss4, metrics_list[3] = self.compute_loss(Z_2, Z_4) 

        loss = (loss1 + loss2 + loss3 + loss4) / 4

        metrics = {}
        for li in metrics_list:
            for key in li.keys():
                metrics[key] = metrics.get(key, 0) + li[key] / 4

        return loss, metrics
    def get_step_loss(self, data, model, scaler, device):
        x, y = data
        
        x = x.to(device)
        y = y.to(device)

        x_1 = x[:, 0, :]
        x_2 = x[:, 1, :]
        x_3 = x[:, 2, :]
        x_4 = x[:, 3, :]

        with autocast(enabled=(scaler is not None)):
            z_1 = model(x_1, training=True)
            z_2 = model(x_2, training=True)
            z_3 = model(x_3, training=True)
            z_4 = model(x_4, training=True)
            loss, metrics = self.compute_loss_MC(z_1, z_2, z_3, z_4)
        
        return loss, metrics