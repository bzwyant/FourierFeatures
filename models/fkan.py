import torch
from torch import nn
import numpy as np


class FourierKANLayer(nn.Module):
    """
    KAN layer with learnable Fourier features on edges
    this is inspired by a the FKAN model (https://github.com/Ali-Meh619/FKAN)
    
    Each edge learns a, b, c from the equation:
    f(x) = a * cos(2 * pi * (bx)) + c * sin(2 * pi * (dx))
    """
    def __init__(self, input_dim, output_dim, grid_size, 
                 include_bias=True, 
                 learnable_freq_scale='off', 
                 smooth_initialization=False,
                 init_freq_scale=1.0):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.include_bias = include_bias
        self.learnable_freq_scale = learnable_freq_scale

        self.norm = nn.LayerNorm(output_dim)  # Normalization layer for output

        # why this normalization factor?
        grid_norm_factor = (torch.arange(grid_size) + 1) ** 2 if smooth_initialization else np.sqrt(grid_size)
        coef_norm_factor = np.sqrt(input_dim) * grid_norm_factor

        self.cos_amplitudes = nn.Parameter(torch.randn(output_dim, input_dim, grid_size) / coef_norm_factor)
        self.sin_amplitudes = nn.Parameter(torch.randn(output_dim, input_dim, grid_size) / coef_norm_factor)

        if self.include_bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))

        assert learnable_freq_scale in ['off', 'connection', 'global'], \
            "learnable_freq_scale must be one of 'off', 'connection', or 'global'"
        
        #TODO: is there a better scale to use when initializing the frequencies?
        init_freqs = torch.arange(init_freq_scale, grid_size * init_freq_scale + init_freq_scale, init_freq_scale)
        if learnable_freq_scale == 'connection':
            # Learnable frequencies for each connection
            self.frequencies = nn.Parameter(init_freqs.repeat(output_dim, input_dim, 1))    # repeat the last dimension 1 time
        elif learnable_freq_scale == 'global':
            # Single learnable frequency for all connections
            self.frequencies = nn.Parameter(init_freqs)
        else:
            self.register_buffer('frequencies', init_freqs)
        

    def forward(self, x):
        x = x.clone().detach().requires_grad_(True)  # Ensure x is differentiable

        x = x.view(-1, self.input_dim)  # Flatten input for batch processing

        # Reshape for broadcasting
        x_expanded = x.view(x.shape[0], 1, x.shape[1], 1)

        if self.learnable_freq_scale == 'connection':
            freqs = self.frequencies.unsqueeze(0)  # (1, output_dim, input_dim, grid_size)
        else:
            freqs = self.frequencies.view(1, 1, 1, self.grid_size)
        
        # Compute trigonometric components
        phases = freqs * x_expanded 
        cos_component = torch.cos(phases)
        sin_component = torch.sin(phases) 

        # Compute the output by summing the contributions from cosine and sine components
        output = torch.sum(cos_component * self.cos_amplitudes.unsqueeze(0), dim=(-2, -1))  # Sum over input_dim and grid_size
        output += torch.sum(sin_component * self.sin_amplitudes.unsqueeze(0), dim=(-2, -1))

        if self.include_bias:
            output += self.bias.unsqueeze(0)  # Add bias if included

        output = output.view(-1, self.output_dim)  # Reshape to match output dimensions
        output = self.norm(output)
        return output
    

    def get_amplitude_phase_freq(self, out_idx, in_idx):
        """
        Get amplitude and phase for a specific output and input index
        """
        a = self.cos_amplitudes[out_idx, in_idx]
        b = self.sin_amplitudes[out_idx, in_idx]

        amplitudes = torch.sqrt(a**2 + b**2)    # has size (grid_size,)
        phases = torch.atan2(b, a)

        return amplitudes, phases
    
    

class LinearINRLayer(nn.Module):
    """
    Linear layer from the FKAN model, which is not discussed in the paper
    """
    def __init__(self, input_dim, output_dim, bias=True, omega=30):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.omega = omega

        # Initialize weights according to the SIREN paper
        with torch.no_grad():
            self.linear.weight.uniform_(-np.sqrt(6 / input_dim) / omega,
                                         np.sqrt(6 / input_dim) / omega)
            
    def forward(self, x):
        linear_output = self.linear(x)
        return (linear_output + torch.tanh(self.omega * linear_output)) * torch.sigmoid(linear_output)


class FourierKAN_INR(nn.Module):
    """
    Fourier KAN INR model
    """
    def __init__(self, input_dim, 
                 output_dim, 
                 hidden_dim, 
                 grid_size, 
                 learnable_freq_scale='off', 
                 smooth_initialization=True):
        super().__init__()

        fkan_layer = FourierKANLayer(input_dim, hidden_dim, grid_size, 
                                include_bias=True, 
                                learnable_freq_scale=learnable_freq_scale,
                                smooth_initialization=smooth_initialization)
        
        linear_layers = [
            LinearINRLayer(hidden_dim, 2 * hidden_dim),
            LinearINRLayer(2 * hidden_dim, 2 * hidden_dim),
            LinearINRLayer(2 * hidden_dim, 2 * hidden_dim),
            LinearINRLayer(2 * hidden_dim, 4 * output_dim),
        ]

        net = [
            fkan_layer,
            *linear_layers,
            nn.Linear(4 * output_dim, output_dim)  # Final linear layer to map to output_dim
        ]

        self.net = nn.Sequential(*net)


    def forward(self, coords):
        output = self.net(coords)
        return output, coords

