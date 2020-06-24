import torch
import torch.nn as nn
from math import pi

class FourierNet(nn.Module):
    def __init__(self, num_layers=4, num_units=256, fourier=True):
        self.num_layers = num_layers
        self.num_units = num_units
        super(FourierNet, self).__init__()

        relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        #fourier option
        if fourier:
            self.B = nn.Linear(2, num_units)
            nn.init.normal_(self.B.weight, std=10.0)
            self.B.weight.requires_grad = False
        layers = []
        for layer in range(self.num_layers):
            if layer == 0:
                layers.append(nn.Linear(2*num_units, num_units))
            elif layer == self.num_layers-1:
                layers.append(nn.Linear(num_units, 3))
            else:
                layers.append(nn.Linear(num_units, num_units))

            if layer != self.num_layers-1:
                layers.append(relu)
            else:
                layers.append(self.sigmoid)

        self.layers = nn.Sequential(*layers)


    def fourier_map(self, x):
        sinside = torch.sin(2 * pi * self.B(x))
        cosside = torch.cos(2 * pi * self.B(x))
        return torch.cat([sinside, cosside], -1)


    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0,2,3,1).view(b, -1, c)
        x = self.fourier_map(x)
        r = self.layers(x)
        return r.view(b, 3, h, w)

