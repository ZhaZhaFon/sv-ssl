from torch import nn

from sslsv.encoders.ThinResNet34 import ThinResNet34
from sslsv.models.byol.MLP import MLP

class onlineNetwork(nn.Module):

    def __init__(self, mlp_dim, enc_dim):
        super().__init__()
        
        self.encoder = ThinResNet34(encoded_dim=enc_dim)
        self.projector = MLP(mlp_dim, 256)
        self.predictor = MLP(256, 256)

    def forward(self, img):
        representation = self.encoder(img)
        projection = self.projector(representation)
        prediction = self.predictor(projection)
        
        return prediction