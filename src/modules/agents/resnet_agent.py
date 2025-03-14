import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class ResNetAgent(nn.Module):
    def __init__(self, input_channel, args):
        super(ResNetAgent, self).__init__()
        self.args = args

        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        self.feature_extractor = self._build_feature_extractor(resnet, "avgpool")

        # with torch.no_grad():
        #     dummy_input = torch.randn(1, 3, 224, 224)
        #     features = self.feature_extractor(dummy_input)
        #     self.feature_dim = features.view(features.size(0), -1).shape[1]
        #     print("Feature Dim: {}".format(self.feature_dim))
        
        self.mlp = nn.Sequential(
            nn.Linear(input_channel, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, args.n_actions)
        )

    def _build_feature_extractor(self, resnet, target_layer):
        layer_dict = {
            'layer1': nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, 
                                    resnet.layer1),
            'layer2': nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, 
                                    resnet.layer1, resnet.layer2),
            'layer3': nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, 
                                    resnet.layer1, resnet.layer2, resnet.layer3),
            'layer4': nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, 
                                    resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4),
            'avgpool': nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, 
                                     resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4, resnet.avgpool),
            'fc': nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, 
                                resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4, resnet.avgpool, resnet.fc)
        }
        return layer_dict[target_layer]
    
    def get_mlp_input(self, x, add_info):
        x = torch.tensor(x, dtype=torch.float32).to("cuda")
        with torch.no_grad():
            x = self.feature_extractor(x)
        x = x.view(x.size(0), -1).cpu().numpy()
        x = np.concatenate([x, add_info], axis=1)
        return x

    def forward(self, x):
        x = self.mlp(x)
        return x


def test():
    from types import SimpleNamespace
    arg = SimpleNamespace()
    arg.rnn_hidden_dim = 128
    arg.n_actions = 9
    model = ResNetAgent(input_channel=3, args=arg)
    state = torch.rand(1, 3, 224, 224)
    out = model.forward(state)
    print(out.shape)

    total_params = sum(p.numel() for p in model.parameters())
    size_bytes = total_params * 4
    size_mb = size_bytes / (1024 ** 2)
    print(f"参数总数: {total_params}, 占用空间: {size_mb:.2f} MB")


if __name__ == "__main__":
    test()

