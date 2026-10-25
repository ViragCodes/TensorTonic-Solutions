import torch

def neuron_forward(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> tuple:
    """
    Returns a tuple of scalar tensors: preactivation and tanh output.
    """
    z = (inputs * weights).sum() + bias
    return (z, z.tanh())
