import torch
import numpy as np


def to_numpy(x):
    return x.detach().cpu().numpy()


def get_mgrid_2D(shape):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1
    shape: tuple/list with desired dimensions of the grid 
    '''
    tensors = tuple([torch.linspace(-1, 1, steps=s) for s in shape])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    mgrid = mgrid.reshape(-1, len(shape))
    return mgrid



def channelwise_norm(img: torch.Tensor):
    c = img.shape[0]
    if c != 3:
        raise ValueError("Image must have 3 channels. in format (C, H, W)")
    
    mean = img.mean(dim=(1, 2))
    std = img.std(dim=(1, 2))

    print("Mean: ", mean)
    print("Std: ", std)
    
    for i in range(c):
        img[i] = (img[i] - mean[i]) / std[i]

    return img, mean, std


def channelwise_denorm(img: torch.Tensor, mean, std):
    c = img.shape[0]
    if c != 3:
        raise ValueError("Image must have 3 channels. in format (C, H, W)")
    
    for i in range(c):
        img[i] = img[i] * std[i] + mean[i]

    return img


def psnr(pred, gt):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between predicted and ground truth images.
    
    Args:
        pred (torch.Tensor): Predicted image tensor.
        gt (torch.Tensor): Ground truth image tensor.
    
    Returns:
        float: PSNR value.
    """
    mse = torch.mean((pred - gt) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))