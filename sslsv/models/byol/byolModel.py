import torch
from torch import nn
import torch.nn.functional as F

from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
#from torchlars import LARS

from sslsv.encoders.ThinResNet34 import ThinResNet34
from sslsv.losses.InfoNCE import InfoNCE
from sslsv.models.byol.onlineNetwork import onlineNetwork
from sslsv.models.byol.targetNetwork import targetNetwork

from torch.cuda.amp import autocast

import copy

class byolModel(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.encoded_dim = 2048
        
        self.online = onlineNetwork(config.mlp_dim, self.encoded_dim)
        self.target = targetNetwork(self.online, config.mlp_dim, self.encoded_dim)

        self.InfoNCE = InfoNCE()

        self.beta = config.beta

        # TODO: increase beta towards 1 during training
        # LR scheduler
        # LARS optimizer

    def forward(self, img1, img2=None, training=False):
        # At the end of training, everything but the online encoder is discarded
        if not training: return self.online.encoder(img1)

        assert img2 is not None,\
            "need 2 inputs for forward when training"

        online_pred = self.online(img1)
        target_proj = self.target(img2)
        
        return online_pred, target_proj
        
    def compute_accuracy(self, online, target):
        N = online.size()[0]
        dot = F.normalize(online, p=2, dim=1) @ F.normalize(target, p=2, dim=1).T
        dot = dot / 0.07
        labels = torch.arange(N, device=online.device)

        pred_indices = torch.argmax(F.softmax(dot, dim=1), dim=1)
        preds_acc = torch.eq(pred_indices, labels)
        accuracy = torch.count_nonzero(preds_acc) / N

        return accuracy

    def get_accuracy(self, Z1, Z2):
        accuracy = self.compute_accuracy(Z1[0], Z1[1])
        accuracy += self.compute_accuracy(Z2[0], Z2[1])
        
        return accuracy / 2

    def compute_loss(self, online, target):
        x = F.normalize(online, dim=-1, p=2)
        y = F.normalize(target, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)        

    def get_loss(self, Z1, Z2):
        loss1 = self.compute_loss(Z1[0], Z1[1])
        loss2 = self.compute_loss(Z2[0], Z2[1])
        
        return (loss1 + loss2).mean()

    def get_metrics(self, loss, accuracy):
        metrics = {}
        metrics['train_loss'] = loss
        metrics['train_accuracy'] = accuracy
        
        return metrics

    def get_step_loss(self, data, scaler, model, device):
        x, y = data
        x = x.to(device)
        y = y.to(device)

        x_1 = x[:, 0, :]
        x_2 = x[:, 1, :]

        with autocast(enabled=(scaler is not None)):
            """
            Z1, Z2 = model(x_1, x_2, training = True)
            loss = self.get_loss(Z1, Z2)
            accuracy = self.get_accuracy(Z1, Z2)
            """
            Z = model(x_1, x_2, training=True)
            loss, accuracy = self.InfoNCE((Z[0], Z[1]))

            metrics = self.get_metrics(loss, accuracy)

        # print(metrics)

        # Update target network
        self.target.EMA_update(self.online, self.beta)

        return loss, metrics