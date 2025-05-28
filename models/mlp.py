import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()
        self.in_features = layer_dims[0]
        self.out_features = layer_dims[-1]

        self.net = []
        self.net.append(nn.Linear(self.in_features, layer_dims[1]))
        self.net.append(nn.ReLU())

        for i in range(1, len(layer_dims)-2):
            self.net.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            self.net.append(nn.BatchNorm1d(layer_dims[i+1]))
            self.net.append(nn.ReLU())

        self.net.append(nn.Linear(layer_dims[-2], self.out_features))
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords
