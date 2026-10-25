import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    x = [p[t] for t, p in zip(y_true, y_pred)]
    return -np.mean(np.log(x))