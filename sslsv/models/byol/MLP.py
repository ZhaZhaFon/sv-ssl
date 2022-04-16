from torch import nn

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

