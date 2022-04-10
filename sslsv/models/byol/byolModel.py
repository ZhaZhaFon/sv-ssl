import torch
from torch import nn
import torch.nn.functional as F

from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
#from torchlars import LARS

from sslsv.encoders.ThinResNet34 import ThinResNet34

from torch.cuda.amp import autocast

import copy

class MLP(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, dim_out)
        )
    def forward(self, X):
        return self.net(X)

class byolModel(nn.Module):
    def __init__(self, config):
        super().__init__()
              
        self.mlp_dim = config.mlp_dim

        self.online_encoder = ThinResNet34(encoded_dim=2048)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.online_projector = MLP(self.mlp_dim, 256)
        self.target_projector = MLP(self.mlp_dim, 256)
        
        self.predictor = MLP(256, 256)

        """
        base_optimizer = SGD(list(self.online_encoder.parameters()) + 
                             list(self.online_projector.parameters()) + 
                             list(self.predictor.parameters()), 
                             lr=config.training.learning_rate,
                             weight_decay=config.training.weight_reg,
                             momentum=0.9)
                             """
        #self.optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)
        # self.optimizer = base_optimizer

        # self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.95)

        self.beta = config.beta
        # TODO: increase beta towards 1 during training

    def forward(self, img1, img2=None, training=False):
        #At the end of training, everything but the online encoder is discarded
        if not training: return self.online_encoder(img1)

        assert img2 is not None,\
            "need 2 inputs for forward when training"

        # online network
        onl_repr_img1 = self.online_encoder(img1)
        onl_repr_img2 = self.online_encoder(img2)
        
        onl_proj_img1 = self.online_projector(onl_repr_img1)
        onl_proj_img2 = self.online_projector(onl_repr_img2)
        
        onl_pred_img1 = self.predictor(onl_proj_img1)
        onl_pred_img2 = self.predictor(onl_proj_img2)

        # target network
        with torch.no_grad(): # do not calculate gradients --> no backprop on target network
            self.update_target_network()
            tar_repr_img1 = self.target_encoder(img1)
            tar_repr_img2 = self.target_encoder(img2)
        
            tar_proj_img1 = self.target_projector(tar_repr_img1)
            tar_proj_img2 = self.target_projector(tar_repr_img2)

            tar_proj_img1.detach()
            tar_proj_img2.detach()

        return (onl_pred_img1, tar_proj_img1), (onl_pred_img2, tar_proj_img2)
    
    # exponential moving average
    def update_target_network(self):
        for tar_params_enc, onl_params_enc in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            tar_params_enc.data = self.beta * tar_params_enc.data + (1 - self.beta) * onl_params_enc.data

        for tar_params_proj, onl_params_proj in zip(self.target_projector.parameters(), self.online_projector.parameters()):
            tar_params_proj.data = self.beta * tar_params_proj.data + (1 - self.beta) * onl_params_proj.data

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
        
        # return (loss1 + loss2).mean()
        # return loss1.mean()
        return loss1.mean() + loss2.mean()

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
            Z1, Z2 = model(x_1, x_2, training = True)
            loss = self.get_loss(Z1, Z2)
            accuracy = self.get_accuracy(Z1, Z2)
            metrics = self.get_metrics(loss, accuracy)

        print(metrics)

        return loss, metrics