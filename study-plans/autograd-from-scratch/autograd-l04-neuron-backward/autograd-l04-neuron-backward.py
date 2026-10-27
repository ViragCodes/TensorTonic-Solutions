import torch

def neuron_backward(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, upstream_gradient: torch.Tensor) -> tuple:
    """
    Returns a tuple of tensors: output, input gradients, weight gradients, bias gradient.
    """
    x = (inputs @ weights) + bias
    y = x.tanh()
    t_grad = upstream_gradient * (1 - y.square())
    i_grad = t_grad * weights
    w_grad = t_grad * inputs
    b_grad = t_grad
    return y, i_grad, w_grad, b_grad 
