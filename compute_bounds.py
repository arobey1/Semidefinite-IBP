import torch
import numpy as np

FNAME = 'mnist.pt'
EPSILON = 0.03

def compute_bounds():

    # model = torch.load('mnist.pt')
    # weights, biases = get_tensors(model)

    weights = [np.random.rand(3, 2), np.random.rand(2, 3)]
    biases = [np.random.rand(3, 1), np.random.rand(2, 1)]


    ones = np.ones((2, 1))
    x = ones * 0.5
    z_lower_0, z_upper_0 = x - EPSILON * ones, x + EPSILON * ones

    z_lower_1, z_upper_1 = affine_layer(z_lower_0, z_upper_0, weights[0], biases[0])
    z_lower_2, z_upper_2 = relu_layer(z_lower_1, z_upper_1)
    z_lower_3, z_upper_3 = affine_layer(z_lower_2, z_upper_2, weights[1], biases[1])

    lowers = [z_lower_0, z_lower_1, z_lower_2, z_lower_3]
    uppers = [z_upper_0, z_upper_1, z_upper_2, z_upper_3]

    return lowers, uppers, weights, biases


def get_tensors(model):
    """Get weight and bias tensors from saved model

    params:
        model: pytorch model

    returns:
        weights: list of np arrays - weights of neural network
        biases: list of np arrays  - biases of neural network
    """

    weights, biases = [], []
    for param_tensor in model.state_dict():
        tensor = model.state_dict()[param_tensor].detach().numpy()
        if 'weight' in param_tensor:
            weights.append(tensor)
        elif 'bias' in param_tensor:
            biases.append(np.expand_dims(tensor, axis=1))

    return weights, biases

def affine_layer(z_lower_km1, z_upper_km1, W, b):
    """Compute interval bounds for affine layer

    params:
        z_lower_km1: (n,1) array - lower bound on previous interval
        z_upper_km1: (n,1) array - upper bound on previous interval
        W: (m,n) array           - weight matrix for affine layer
        b: (m,1) array           - bias vector for affine layer

    returns:
        z_lower_k: (m,1) array - lower bound on current interval
        z_upper_k: (m,1) array - upper bound on current interval
    """

    mu_km1 = 0.5 * (z_upper_km1 + z_lower_km1)
    r_km1 = 0.5 * (z_upper_km1 - z_lower_km1)
    mu_k = W @ mu_km1 + b
    r_k = abs(W) @ r_km1

    z_lower_k = mu_k - r_k
    z_upper_k = mu_k + r_k

    return z_lower_k, z_upper_k

def relu_layer(z_lower_km1, z_upper_km1):
    """Compute interval bounds for ReLU nonlinearity

    params:
        z_lower_km1: (n,1) array - lower bound on previous interval
        z_upper_km1: (n,1) array - upper bound on previous interval

    returns:
        z_lower_k: (m,1) array - lower bound on current interval
        z_upper_k: (m,1) array - upper bound on current interval
    """

    z_lower_k = np.maximum(0, z_lower_km1)
    z_upper_k = np.maximum(0, z_upper_km1)

    return z_lower_k, z_upper_k
