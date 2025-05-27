import torch
from torch import nn


class FourierFeaturesKANLayer(nn.Module):
    """
    KAN layer with learnable Fourier features on edges
    
    Each edge learns a, b, c from the equation:
    f(x) = a * cos(2 * pi * (b^T @ x)) + c * sin(2 * pi * (b^T @ x))
    
    where:
    """
    def __init__(self, input_dim, output_dim, num_frequencies=10, freq_init_scale=1.0, bias=True):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_frequencies = num_frequencies

        self.frequencies = nn.Parameter(
            torch.randn(output_dim, input_dim, num_frequencies) * freq_init_scale
        )

        self.cos_weights = nn.Parameter(torch.randn(output_dim, input_dim, num_frequencies))
        self.sin_weights = nn.Parameter(torch.randn(output_dim, input_dim, num_frequencies))

        self.bias = bias
        if bias:
            self.bias_terms = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        # batch_size = x.shape[0]

        # Expand input to match the frequency dimensions
        # shape: (batch_size, 1, input_dim, 1)
        x_expanded = x.unsqueeze(1).unsqueeze(-1)

        # compute the frequency terms: 2 * np.pi * b_i^T * x
        freq_terms = 2 * torch.pi * (x_expanded * self.frequencies)

        # compute cosine and sine components
        cos_terms = self.cos_weights * torch.cos(freq_terms)
        sin_terms = self.sin_weights * torch.sin(freq_terms)
        # sum over frequencies
        fourier_features = cos_terms + sin_terms

        # shape: (batch_size, output_dim)
        output = torch.sum(fourier_features, dim=(-2, -1))
        if self.bias:
            output += self.bias_terms

        return output

    def get_amplitude_phase(self, out_idx, in_idx):
        """
        Get amplitude and phase for a specific output and input index
        """
        a = self.cos_weights[out_idx, in_idx]
        c = self.sin_weights[out_idx, in_idx]

        amplitude = torch.sqrt(a**2 + c**2)
        phase = torch.atan2(a, c)
        frequencies = self.frequencies[out_idx, in_idx]

        return amplitude, phase, frequencies
    

class FourierFeauturesKAN(nn.Module):
    """
    KAN network with Fourier feature layers.
    """
    def __init__(self, layer_dims, num_frequencies=10, freq_init_scale=1.0, bias=True):
        """
        Args:
            layer_dims (list): List of dimensions for each layer.
            num_frequencies (int): Number of frequencies to learn for each edge.
            freq_init_scale (float): Scale for initializing frequency parameters.
            bias (bool): Whether to include bias terms in the layers.
        """
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            layer = FourierFeaturesKANLayer(
                input_dim=layer_dims[i],
                output_dim=layer_dims[i + 1],
                num_frequencies=num_frequencies,
                freq_init_scale=freq_init_scale,
                bias=bias
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x