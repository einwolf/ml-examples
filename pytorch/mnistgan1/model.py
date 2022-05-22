import torch.nn as nn


class discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ) # yapf: disable

    def forward(self, input):
        return self.main(input)


class generator(nn.Module):

    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
        ) # yapf: disable

    def forward(self, input):
        return self.main(input)
