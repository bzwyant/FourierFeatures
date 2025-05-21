import torch
import torch.nn.functional as F


def image_mse(model_output, gt):
    """
    Computes the Mean Squared Error (MSE) loss between the model output and ground truth images.
    
    Args:
        model_output (torch.Tensor): The output from the model, shape (batch_size, channels, height, width).
        gt (torch.Tensor): The ground truth images, shape (batch_size, channels, height, width).
    
    Returns:
        torch.Tensor: The computed MSE loss.
    """
    return ((model_output - gt) ** 2).mean()