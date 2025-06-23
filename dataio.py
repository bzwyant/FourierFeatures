import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from PIL import Image
from utils import get_mgrid_2D


class ImageFitting(Dataset):
    def __init__(self, filename, normalization=True, size=256):
        super().__init__()
        self.pil_img = Image.open(filename)
        self.normalization = normalization
        self.original_size = (self.pil_img.width, self.pil_img.height)

        # Convert to tensor and get dimensions
        rgb_img = self.pil_img.convert('RGB')

        transform_list = []
        if size is not None:
            transform_list.append(Resize(size))
        transform_list.append(ToTensor())
        transform = Compose(transform_list)

        self.img_tensor = transform(rgb_img)  # Changes from (H, W, C) to (C, H, W)

        self.channels, self.H, self.W = self.img_tensor.shape
        
        # Calculate statistics
        self.original_mean = self.img_tensor.mean(dim=(1, 2))
        self.original_std = self.img_tensor.std(dim=(1, 2))

        # Normalize
        if normalization:
            normalize = Normalize(torch.tensor([0.5]), torch.tensor([0.5]))
            self.normalized_tensor = normalize(self.img_tensor)
        else:
            self.normalized_tensor = self.img_tensor
        self.normalized_tensor = self.normalized_tensor.permute(1, 2, 0)  # back to (H, W, C)

        # Prepare for efficient access
        self.pixels = self.normalized_tensor.reshape(-1, self.channels)
        self.coords = get_mgrid_2D((self.H, self.W))

        # Precompute indices for efficient random access
        self.num_pixels = self.H * self.W
        self.all_indices = torch.arange(self.num_pixels)

        assert len(self.pixels) == len(self.coords), "The number of pixels and coordinates don't match."

    def to(self, device):
        """Move tensors to specified device"""
        self.device = device
        self.pixels = self.pixels.to(device)
        self.coords = self.coords.to(device)
        self.all_indices = self.all_indices.to(device)
        return self

    def get_batch_indices(self, batch_size, random=True):
        """
        Returns a batch of shuffled indices for efficient training
        """
        if random:
            indices = torch.randperm(self.num_pixels)
        else:
            indices = self.all_indices

        batches = []
        for b_idx in range(0, self.num_pixels, batch_size):
            batches.append(indices[b_idx:min(self.num_pixels, b_idx+batch_size)])

        return batches

    def get_items_by_indices(self, indices):
        """
        Returns coordinates and pixels for given indices
        """
        return {
            'indices': indices,
            'coords': self.coords[indices],
            'pixels': self.pixels[indices]
        }

    def create_reconstruction_tensor(self):
        """
        Creates an empty tensor for reconstructed image
        """
        return torch.zeros_like(self.pixels)

    def reconstruction_to_image(self, reconstructed):
        """
        Converts a flat reconstructed tensor to an image
        """
        return reconstructed.reshape(self.H, self.W, self.channels)

    def __len__(self):
        return self.num_pixels

    def __getitem__(self, idx):
        # This still works for traditional dataloader usage if needed
        return {'idx': idx, 'coords': self.coords[idx], 'pixels': self.pixels[idx]}

    def denormalize(self, normalized_tensor):
        if not self.normalization:
            return normalized_tensor

        if len(normalized_tensor.shape) == 3:  # Already in image shape
            tensor_to_denorm = normalized_tensor.permute(2, 0, 1)
        else:
            tensor_to_denorm = normalized_tensor.reshape(self.H, self.W, self.channels).permute(2, 0, 1)

        denormalized_tensor = tensor_to_denorm * 0.5 + 0.5
        denormalized_tensor = torch.clamp(denormalized_tensor, 0, 1)

        return denormalized_tensor.permute(1, 2, 0)  # Back to H, W, C
    

class EncodedImageFitting(ImageFitting):
    """
    Using random Fourier features approach from https://bmild.github.io/fourfeat/

    Extension of ImageFitting with flexible coordinate encoding options:
    1. Basic sinusoidal encoding - maps coordinates to [cos(2πv), sin(2πv)]
    2. Gaussian random Fourier features
    """
    def __init__(self, filename, normalization=False, encoding_type="gaussian", 
                 embedding_size=256, scale=15, include_original=False, seed=42,
                 shuffle=False, channels=None):
        super().__init__(filename, normalization)
        
        # Store parameters
        self.encoding_type = encoding_type.lower()
        self.scale = scale
        self.include_original = include_original
        
        if self.encoding_type not in ["basic", "gaussian"]:
            raise ValueError("encoding_type must be 'basic' or 'gaussian'")
        
        # For Gaussian encoding
        if self.encoding_type == "gaussian":
            self.embedding_size = embedding_size
            # Set random seed for reproducibility
            torch.manual_seed(seed)
            # Generate random Gaussian Fourier basis
            self.B = torch.randn(embedding_size, 2) * scale
        else:
            self.embedding_size = None
            self.B = None

        # For test with shuffled coordinates (should not work in practice)
        self.shuffle = shuffle
        
        # Create the positional encoding for coordinates
        self.encoded_coords = self._encode_coordinates(self.coords)
    
    def _basic_encoding(self, coords):
        """
        Apply basic sinusoidal encoding to coordinates: [cos(2πv), sin(2πv)]
        
        Args:
            coords: Tensor of shape [N, 2] containing x,y coordinates
        """
        cos_features = torch.cos(2 * torch.pi * coords * self.scale)
        sin_features = torch.sin(2 * torch.pi * coords * self.scale)
        
        encoded_list = []
        
        if self.include_original:
            encoded_list.append(coords)
        
        encoded_list.append(cos_features)
        encoded_list.append(sin_features)
        
        encoded_coords = torch.cat(encoded_list, dim=1)
        
        return encoded_coords
    
    def _gaussian_encoding(self, coords):
        """
        Apply Gaussian random Fourier feature encoding to coordinates.
        
        Args:
            coords: Tensor of shape [N, 2] containing x,y coordinates
        """
        # Project coordinates onto random Gaussian directions (B)
        projection = coords @ self.B.t()
        
        # Apply sine and cosine to get Fourier features
        sin_features = torch.sin(projection)
        cos_features = torch.cos(projection)
        
        encoded_list = []
        
        # Include original coordinates if specified
        if self.include_original:
            encoded_list.append(coords)
        
        encoded_list.append(sin_features)
        encoded_list.append(cos_features)
        
        encoded_coords = torch.cat(encoded_list, dim=1)
        
        return encoded_coords
    
    def _encode_coordinates(self, coords):
        """
        Apply the selected encoding method to coordinates.
        """
        if self.encoding_type == "basic":
            encoded_coords = self._basic_encoding(coords)
        else:  # gaussian
            encoded_coords = self._gaussian_encoding(coords)

        if self.shuffle:
            encoded_coords = encoded_coords[torch.randperm(encoded_coords.shape[0])]
        
        return encoded_coords
    
    def to(self, device):
        """Move tensors to specified device, extending the parent method"""
        super().to(device)
        if self.B is not None:
            self.B = self.B.to(device)
        self.encoded_coords = self.encoded_coords.to(device)
        return self
    
    def get_items_by_indices(self, indices):
        """
        Returns encoded coordinates and pixels for given indices,
        overriding the parent method
        """
        if self.channels is not None:
            assert self.channels <= self.pixels.shape[1], "Number of channels exceeds available channels."
            pixels = self.pixels[indices][self.channels]

        return {
            'indices': indices,
            'coords': self.encoded_coords[indices],  # Original coords for reference
            # 'encoded_coords': self.encoded_coords[indices],  # Encoded coords
            'pixels': pixels if self.channels is not None else self.pixels[indices]
        }
    
    def __getitem__(self, idx):
        """Override to return encoded coordinates"""
        base_item = super().__getitem__(idx)
        base_item['coords'] = self.encoded_coords[idx]
        return base_item
    
    @property
    def encoding_dim(self):
        """Return the dimension of the encoded coordinates"""
        return self.encoded_coords.shape[1]
    
    def get_encoding_info(self):
        """Return information about the encoding"""        
        info = {
            'encoding_type': self.encoding_type,
            'include_original': self.include_original,
            'scale': self.scale,
            'original_dim': 2,  # x, y coordinates
            'encoded_dim': self.encoding_dim
        }
        
        if self.encoding_type == "gaussian":
            info.update({
                'embedding_size': self.embedding_size,
                'sin_cos_features': self.embedding_size * 2,
            })
        else: 
            info.update({
                'sin_cos_features': 4,  # 2 coords x 2 features (sin, cos)
            })
            
        return info