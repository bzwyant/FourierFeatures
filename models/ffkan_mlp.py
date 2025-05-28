from .fourier_kan import FfKan
from .mlp import MLP

import torch
import torch.nn as nn


class FfKanMLP(nn.Module):
    """
    A few FFKAN layers followed by a standard MLP.
    """
    def __init__(self, ffkan_layer_dims, mlp_layer_dims,
                 num_frequencies=10, freq_init_scale=1.0, bias=True):
        super().__init__()
        if ffkan_layer_dims[-1] != mlp_layer_dims[0]:
            raise ValueError("The last layer of FFKAN must match the first layer of MLP in terms of output features.")
        if len(ffkan_layer_dims) < 2:
            raise ValueError("ffkan_layer_dims must be a list with 2 or more integers")
        if len(mlp_layer_dims) < 2:
            raise ValueError("mlp_layer_dims must be a list with 2 or more integers")
        
        self.fourier_kan_block = FfKan(
            layer_dims=ffkan_layer_dims,
            num_frequencies=num_frequencies,
            freq_init_scale=freq_init_scale,
            bias=bias
        )
        self.mlp_block = MLP(
            in_features=ffkan_layer_dims[-1],
            out_features=mlp_layer_dims[-1],
            layer_dims=mlp_layer_dims
        )

    
def forward(self, x):
    coords = x.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
    output, _ = self.fourier_kan_block(x)
    output, _ = self.mlp_block(output) 
    return output, coords
