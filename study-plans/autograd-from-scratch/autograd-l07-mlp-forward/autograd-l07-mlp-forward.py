import torch

def mlp_forward(inputs: torch.Tensor, weights: list[torch.Tensor], biases: list[torch.Tensor]) -> tuple:
    """
    Returns the final output tensor and a list of layer-output tensors.
    """
    l = [inputs]
    for w, b in zip(weights, biases):
        y = (w @ l[-1] + b).tanh()
        l.append(y)
    return l[-1], l[1:]