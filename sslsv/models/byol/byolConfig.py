from dataclasses import dataclass

from sslsv.configs import ModelConfig

@dataclass
class byolConfig(ModelConfig):
    mlp_dim: int = 2048
    beta: float = 0.996