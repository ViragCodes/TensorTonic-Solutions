import torch

def squared_error_loss_gradients(predictions: torch.Tensor, targets: torch.Tensor) -> tuple:
    """
    Returns a scalar loss tensor and a prediction-shaped gradient tensor.
    """
    return (predictions - targets).square().sum(), 2 * (predictions - targets)
