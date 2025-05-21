import torch
import torch.nn as nn
import numpy as np
from siren import SineLayer

class HybridSiren(nn.Module):
    """A network with specified layers as SIREN layers and others as standard layers."""
    
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30., siren_layers=None, standard_activation=nn.ReLU()):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.outermost_linear = outermost_linear
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0
        self.siren_layers = siren_layers
        self.standard_activation = standard_activation
        
        # Default to first layer being SIREN if not specified
        if siren_layers is None:
            siren_layers = [0]
        
        if not siren_layers:
            # Throw an error because this isn't a hybrid network
            raise ValueError("Siren layers cannot be empty - make a different class for this")
        
        self.net = []
        # First layer
        if 0 in siren_layers:
            self.net.append(SineLayer(in_features, hidden_features,
                                      is_first=True, omega_0=first_omega_0))
        else:
            self.net.append(nn.Linear(in_features, hidden_features))
            self.net.append(nn.BatchNorm1d(hidden_features))
            self.net.append(standard_activation)
        
        # Hidden layers
        for i in range(hidden_layers):
            layer_idx = i + 1  # +1 because we already added the first layer
            if layer_idx in siren_layers:
                self.net.append(SineLayer(hidden_features, hidden_features,
                                          is_first=False, omega_0=hidden_omega_0))
            else:
                self.net.append(nn.Linear(hidden_features, hidden_features))
                self.net.append(nn.BatchNorm1d(hidden_features))
                self.net.append(standard_activation)
        
        # Output layer
        last_layer_idx = hidden_layers + 1
        if last_layer_idx in siren_layers and not outermost_linear:
            # Use SIREN for output layer
            self.net.append(SineLayer(hidden_features, out_features,
                                     is_first=False, omega_0=hidden_omega_0))
        else:
            # Use linear output layer (with SIREN initialization if needed)
            final_linear = nn.Linear(hidden_features, out_features)
            
            if outermost_linear:
                # Initialize with SIREN-appropriate initialization
                with torch.no_grad():
                    final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                                np.sqrt(6 / hidden_features) / hidden_omega_0)
            
            self.net.append(final_linear)
        
        # Convert list of layers to Sequential module
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        """Forward pass through the network."""
        output = self.net(x)
        return output, None  # Return None as second output for compatibility with other models

    def copy(self):
        """Create a copy of the model with the same parameters."""
        model_copy = HybridSiren(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            hidden_layers=self.hidden_layers,
            out_features=self.out_features,
            outermost_linear=self.outermost_linear,
            first_omega_0=self.first_omega_0,
            hidden_omega_0=self.hidden_omega_0,
            siren_layers=self.siren_layers,
            standard_activation=self.standard_activation
        )
        
        # Copy weights and biases
        model_copy.load_state_dict(self.state_dict())
        
        return model_copy