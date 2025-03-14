REGISTRY = {}

from .rnn_agent import RNNAgent
from .cnn_agent import CNNAgent
from .resnet_agent import ResNetAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["cnn"] = CNNAgent
REGISTRY["resnet"] = ResNetAgent