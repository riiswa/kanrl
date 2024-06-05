import torch
import torch.nn as nn

from kan import KAN

# Default parameters for KAN regularization term
LAMB_L1 = 1.0
LAMB_ENTROPY = 2.0
LAMB_COEF = 0.0
LAMB_COEFDIFF = 0.0
SMALL_MAG_THRESHOLD = 1e-16
SMALL_REG_FACTOR = 1.0


# TODO : Used **kwargs to handle the fact MLPs and KANs use diff args, but env and **kwargs doesn't seem clean
def initialize_network(input_size, output_size, **kwargs):
    """Initialize a network with the specified config
    """
    method = kwargs["method"]
    width = kwargs["width"]

    # TODO : Should maybe allow having deeper networks in this function
    if method == "MLP":
        network = nn.Sequential(
            nn.Linear(input_size, width),
            nn.ReLU(),
            nn.Linear(width, output_size),
        )
    elif method == "KAN":
        grid = kwargs["grid"]
        network = KAN(
            width=[input_size, width, output_size],
            grid=grid,
            # TODO : Do you wanna keep these parameters ? 
            k=3,
            bias_trainable=False,
            sp_trainable=False,
            sb_trainable=False,
        )
    else:
        raise Exception(
            f"Method {method} doesn't exist, choose between MLP and KAN."
        )
    return network


def reg(
    net,
    lamb_l1=LAMB_L1,
    lamb_entropy=LAMB_ENTROPY,
    lamb_coef=LAMB_COEF,
    lamb_coefdiff=LAMB_COEFDIFF,
    small_mag_threshold=SMALL_MAG_THRESHOLD,
    small_reg_factor=SMALL_REG_FACTOR
):
    """Compute a regularization term to add it to the current loss
    """
    acts_scale = net.acts_scale
    def nonlinear(x, th=small_mag_threshold, factor=small_reg_factor):
        return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)

    reg_ = 0.0
    for i in range(len(acts_scale)):
        vec = acts_scale[i].reshape(
            -1,
        )

        p = vec / torch.sum(vec)
        l1 = torch.sum(nonlinear(vec))
        entropy = -torch.sum(p * torch.log2(p + 1e-4))
        reg_ += lamb_l1 * l1 + lamb_entropy * entropy  # both l1 and entropy

    # regularize coefficient to encourage spline to be zero
    for i in range(len(net.act_fun)):
        coeff_l1 = torch.sum(torch.mean(torch.abs(net.act_fun[i].coef), dim=1))
        coeff_diff_l1 = torch.sum(
            torch.mean(torch.abs(torch.diff(net.act_fun[i].coef)), dim=1)
        )
        reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

    return reg_