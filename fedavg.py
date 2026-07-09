import numpy as np
import torch
from typing import List, Dict, Optional

def fedavg(weights_list: List[Dict[str, np.ndarray]], 
           sample_sizes: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
    """
    Federated averaging of model weights.
    Args:
        weights_list: List of state dictionaries from clients (numpy arrays).
        sample_sizes: List of sample counts for each client. If None, uses equal weights.
    Returns:
        Dictionary of averaged weights.
    """
    if not weights_list:
        raise ValueError("Empty weights list")
    
    n_clients = len(weights_list)

    # Use equal weights if sample_sizes not provided
    if sample_sizes is None:
        sample_sizes = [1] * n_clients
    elif len(sample_sizes) != n_clients:
        raise ValueError("Mismatched number of weights and sample sizes")

    # Convert all client weights to float32 to avoid dtype conflicts
    weights_list_float = []
    for client_weights in weights_list:
        float_weights = {}
        for k, v in client_weights.items():
            float_weights[k] = v.astype(np.float32)
        weights_list_float.append(float_weights)
    weights_list = weights_list_float

    # We will ignore sample sizes and perform equal averaging across clients

    # Initialize global weights as zeros (float32) using first client's weights as template
    global_weights = {}
    first_client = weights_list[0]
    for k, v in first_client.items():
        global_weights[k] = np.zeros_like(v, dtype=np.float32)

    # Equal-average all parameters including BatchNorm running stats
    bn_keys = ["running_mean", "running_var", "num_batches_tracked"]

    # Aggregate weights
    for key in global_weights:
        # Equal averaging for all keys (parameters and BN buffers)
        for client_weights in weights_list:
            global_weights[key] += client_weights[key] / n_clients
        # For BN counters, keep integral semantics by rounding
        if "num_batches_tracked" in key:
            global_weights[key] = np.round(global_weights[key]).astype(np.int64).astype(np.float32)

    return global_weights


def get_model_weights(model: torch.nn.Module) -> Dict[str, np.ndarray]:
    """
    Extract model parameters and buffers as numpy arrays.
    """
    weights = {}
    # Include model parameters
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy()
    # Include buffers (e.g., BatchNorm running stats)
    for name, buf in model.named_buffers():
        weights[name] = buf.detach().cpu().numpy()
    return weights


def set_model_weights(model, weights, device=None):
    for name, param in model.named_parameters():
        if name in weights:
            weight_tensor = torch.from_numpy(
                weights[name] if isinstance(weights[name], np.ndarray)
                else np.array(weights[name], dtype=np.float32)
            )
            if device is not None:
                weight_tensor = weight_tensor.to(device)
            param.data.copy_(weight_tensor)

    for name, buf in model.named_buffers():
        if name in weights:
            weight_value = weights[name]
            # Accept numpy arrays and numpy/python scalars
            if isinstance(weight_value, np.ndarray):
                weight_tensor = torch.from_numpy(weight_value)
            else:
                # numpy scalar (e.g., np.float32, np.int64) or python scalar
                if np.isscalar(weight_value):
                    weight_tensor = torch.tensor(weight_value)
                else:
                    raise TypeError(f"Unsupported type for buffer {name}: {type(weight_value)}")

            # Ensure dtype/device match target buffer
            if device is not None:
                weight_tensor = weight_tensor.to(device)
            weight_tensor = weight_tensor.to(dtype=buf.dtype)
            buf.data.copy_(weight_tensor)
    return model