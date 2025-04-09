"""
Neural network utilities for reinforcement learning algorithms.

This module provides functions to create and configure different types of neural networks,
primarily MLPs and KANs (Kolmogorov-Arnold Networks), along with regularization utilities
specifically for KAN networks.
"""
import typing as tp
from typing import List, Union, Optional, Literal, Dict, Any, Tuple, Callable

import torch
import torch.nn as nn
from torch import Tensor

from kan import KAN
from .efficient_kan import EfficientKAN


# KAN regularization term
LAMB_L1: float = 1.0
LAMB_ENTROPY: float = 2.0
LAMB_COEF: float = 0.0
LAMB_COEFDIFF: float = 0.0
SMALL_MAG_THRESHOLD: float = 1e-16
SMALL_REG_FACTOR: float = 1.0

NetworkType = Literal["MLP", "KAN", "EFFKAN"]
ActivationType = Callable[[], nn.Module]


def create_mlp(
    input_size: int,
    output_size: int,
    hidden_layers: List[int] = [64],
    activation: ActivationType = nn.ReLU,
) -> nn.Sequential:
    """
    Create a Multi-Layer Perceptron (MLP) with configurable architecture.
    
    Args:
        input_size: Dimension of the input features
        output_size: Dimension of the output
        hidden_layers: List containing the size of each hidden layer
        activation: Activation function to use between layers
        
    Returns:
        A PyTorch Sequential module implementing the MLP
    """
    layers = []
    prev_size = input_size
    
    for size in hidden_layers:
        layers.append(nn.Linear(prev_size, size))
        layers.append(activation())
        prev_size = size
    
    layers.append(nn.Linear(prev_size, output_size))
    return nn.Sequential(*layers)


def create_kan(
    input_size: int,
    output_size: int,
    hidden_layers: List[int] = [64],
    grid: int = 10,
    k: int = 3,
    bias_trainable: bool = False,
    sp_trainable: bool = False,
    sb_trainable: bool = False,
) -> KAN:
    """
    Create a Kolmogorov-Arnold Network (KAN) with configurable architecture.
    
    Args:
        input_size: Dimension of the input features
        output_size: Dimension of the output
        hidden_layers: List containing the size of each hidden layer
        grid: Number of grid points for the KAN
        k: Order of spline interpolation
        bias_trainable: Whether the bias term is trainable
        sp_trainable: Whether the support points are trainable
        sb_trainable: Whether the support base is trainable
        
    Returns:
        A KAN module
    """
    width = [input_size, *hidden_layers, output_size]
    
    return KAN(
        width=width,
        grid=grid,
        k=k,
        bias_trainable=bias_trainable,
        sp_trainable=sp_trainable,
        sb_trainable=sb_trainable,
    )


def create_effkan(
    input_size: int,
    output_size: int,
    hidden_layers: List[int] = [64],
    grid: int = 5,
    k: int = 3,
    scale_noise: float = 0.1,
    scale_base: float = 1.0,
    scale_spline: float = 1.0,
    base_activation: Callable[[], nn.Module] = nn.SiLU,
    grid_eps: float = 0.02,
    grid_range: List[float] = [-1, 1],
) -> EfficientKAN:
    """
    Create an Efficient Kolmogorov-Arnold Network (EFFKAN) with configurable architecture.
    
    Args:
        input_size: Dimension of the input features
        output_size: Dimension of the output
        hidden_layers: List containing the size of each hidden layer
        grid: Number of grid points for the KAN (grid_size parameter)
        k: Order of spline interpolation (spline_order parameter)
        scale_noise: Scale of the initialization noise
        scale_base: Scale for the base activation
        scale_spline: Scale for the spline component
        base_activation: Activation function to use for base component
        grid_eps: Parameter controlling the mixture of uniform and adaptive grid
        grid_range: Range for the grid points [min, max]
        
    Returns:
        An EfficientKAN module
    """
    # Construct the layer dimensions
    layers_hidden = [input_size, *hidden_layers, output_size]
    
    return EfficientKAN(
        layers_hidden=layers_hidden,
        grid_size=grid,
        spline_order=k,
        scale_noise=scale_noise,
        scale_base=scale_base,
        scale_spline=scale_spline,
        base_activation=base_activation,
        grid_eps=grid_eps,
        grid_range=grid_range,
    )


def initialize_network(
    input_size: int,
    output_size: int,
    method: NetworkType = "MLP",
    hidden_layers: Optional[List[int]] = None,
    activation: ActivationType = nn.ReLU,
    grid: Optional[int] = None,
    k: int = 3,
    bias_trainable: bool = False,
    sp_trainable: bool = False,
    sb_trainable: bool = False,
) -> Union[nn.Sequential, KAN]:
    """
    Initialize a neural network with the specified configuration.
    
    Args:
        input_size: Dimension of input features
        output_size: Dimension of output
        method: Network type ("MLP" or "KAN")
        hidden_layers: List of hidden layer sizes (default: [64])
        activation: Activation function (for MLP networks)
        grid: Grid parameter for KAN networks (required if method="KAN")
        k: Order of spline interpolation for KAN
        bias_trainable: Whether the bias term is trainable (KAN only)
        sp_trainable: Whether the support points are trainable (KAN only)
        sb_trainable: Whether the support base is trainable (KAN only)
        
    Returns:
        A PyTorch neural network module
        
    Raises:
        ValueError: If an unsupported network type is specified or required parameters are missing
    """
    if hidden_layers is None:
        hidden_layers = [64]
    
    if method == "MLP":
        return create_mlp(input_size, output_size, hidden_layers, activation)
    elif method == "KAN":
        if grid is None:
            raise ValueError("Grid parameter is required for KAN networks")
        return create_kan(
            input_size, 
            output_size, 
            hidden_layers, 
            grid, 
            k,
            bias_trainable, 
            sp_trainable, 
            sb_trainable
        )
    elif method == "EFFKAN":
        if grid is None:
            raise ValueError("Grid parameter is required for EFFKAN networks")
        return create_effkan(
            input_size,
            output_size,
            hidden_layers,
            grid,
            k,
        )
    else:
        raise ValueError(f"Method {method} doesn't exist, choose between MLP and KAN.")


def reg(
    net: KAN,
    lamb_l1: float = LAMB_L1,
    lamb_entropy: float = LAMB_ENTROPY,
    lamb_coef: float = LAMB_COEF,
    lamb_coefdiff: float = LAMB_COEFDIFF,
    small_mag_threshold: float = SMALL_MAG_THRESHOLD,
    small_reg_factor: float = SMALL_REG_FACTOR
) -> Tensor:
    """
    Compute a regularization term for KAN networks to add to the loss function.
    
    This regularization combines multiple terms:
    1. L1 regularization on activation scales
    2. Entropy regularization on activation scales
    3. L1 regularization on spline coefficients
    4. L1 regularization on differences between spline coefficients
    
    Args:
        net: A KAN network instance
        lamb_l1: Weight for L1 regularization on activation scales
        lamb_entropy: Weight for entropy regularization on activation scales
        lamb_coef: Weight for L1 regularization on coefficients
        lamb_coefdiff: Weight for coefficient difference regularization
        small_mag_threshold: Threshold for nonlinear regularization
        small_reg_factor: Factor for nonlinear regularization
        
    Returns:
        A tensor containing the regularization value
        
    Raises:
        TypeError: If net is not a KAN network
    """
    # Type checking to prevent misuse
    if not hasattr(net, 'acts_scale') or not hasattr(net, 'act_fun'):
        raise TypeError("reg function only works with KAN networks")
    
    # Helper function for nonlinear regularization
    def nonlinear(x: Tensor, th: float = small_mag_threshold, factor: float = small_reg_factor) -> Tensor:
        """Apply nonlinear regularization to tensor elements."""
        return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)
    
    reg_value = 0.0
    
    # Regularize activation scales
    for i in range(len(net.acts_scale)):
        vec = net.acts_scale[i].reshape(-1)
        
        # Avoid division by zero
        sum_vec = torch.sum(vec)
        if sum_vec > 0:
            p = vec / sum_vec
            l1 = torch.sum(nonlinear(vec))
            # Add small constant to avoid log(0)
            entropy = -torch.sum(p * torch.log2(p + 1e-4))
            reg_value += lamb_l1 * l1 + lamb_entropy * entropy

    # Regularize coefficients to encourage spline to be zero
    for i in range(len(net.act_fun)):
        coeff_l1 = torch.sum(torch.mean(torch.abs(net.act_fun[i].coef), dim=1))
        coeff_diff_l1 = torch.sum(
            torch.mean(torch.abs(torch.diff(net.act_fun[i].coef)), dim=1)
        )
        reg_value += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1
    
    return reg_value


# For backward compatibility with existing code
def create_network(*args, **kwargs):
    """Alias for initialize_network for backward compatibility."""
    import warnings
    warnings.warn(
        "create_network is deprecated, use initialize_network instead",
        DeprecationWarning, 
        stacklevel=2
    )
    return initialize_network(*args, **kwargs)