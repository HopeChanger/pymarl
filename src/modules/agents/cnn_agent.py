import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNAgent(nn.Module):
    def __init__(self, input_channel, args):
        super(CNNAgent, self).__init__()
        self.args = args

        self.conv1 = nn.Conv2d(input_channel, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        self.pool = nn.AdaptiveAvgPool2d((8, 8))

        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, args.n_actions)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def test():
    from types import SimpleNamespace
    arg = SimpleNamespace()
    arg.rnn_hidden_dim = 128
    arg.n_actions = 9
    net = CNNAgent(input_channel=3, args=arg)
    state = torch.rand(1, 3, 640, 480)
    out = net.forward(state)
    print(out.shape)


if __name__ == "__main__":
    test()

