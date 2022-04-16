import torch
from torch import nn

from sslsv.encoders.ThinResNet34 import ThinResNet34
from sslsv.models.byol.MLP import MLP
import copy

class targetNetwork(nn.Module):

    def __init__(self, online, mlp_dim, enc_dim):
        super().__init__()
        
        self.encoder = ThinResNet34(encoded_dim=enc_dim)
        self.encoder.load_state_dict(online.encoder.state_dict())

        #self.encoder = copy.deepcopy(online.encoder)
        
        self.projector = MLP(mlp_dim, 256)
        self.projector.load_state_dict(online.projector.state_dict())
        
        for e, p in zip(self.encoder.parameters(), self.projector.parameters()):
            e.requires_grad, p.requires_grad = False, False

    def forward(self, img):
        with torch.no_grad():
            representation = self.encoder(img)
            projection = self.projector(representation)
            projection.detach()
            
        return projection

    def EMA_update(self, online, beta):
        for tar_params_enc, onl_params_enc in zip(self.encoder.parameters(), online.encoder.parameters()):
            tar_params_enc.data = beta * tar_params_enc.data + (1 - beta) * onl_params_enc.data

        for tar_params_proj, onl_params_proj in zip(self.projector.parameters(), online.projector.parameters()):
            tar_params_proj.data = beta * tar_params_proj.data + (1 - beta) * onl_params_proj.data