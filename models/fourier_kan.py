from torch import nn


class FourierKan(nn.Module):
    """Fourier-Kan network for implicit neural representation."""
    def __init__(self):
        super().__init__()