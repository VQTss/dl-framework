import numpy as np
import math


# Rectified Linear Unit (RELU) Activation Function
def relu(x):
    return np.maximum(0, x)

# Sigmoid Activation Function
def sigmoid(x):
    x = x.asarray(x, dtype=float)
    if not isinstance(x, [list, tuple]):
        return 1 / (1 + np.exp(-x))
    return [1 /  (1 + np.exp(-i) ) for i in x ]

# Tanh activation
def tanh(x) -> np.ndarray:
    x = np.array(x)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)) 

# Softmax activation function
def softmax(x):
    """
    Softmax activation function for 2D arrays and 1D arrays
    """
    x = np.array(x, dtype=float)
    if x.ndim == 1:
        x_shift = x - x.max()
        x_exp = np.exp(x_shift)
        return x_exp / x_exp.sum()
    elif x.ndim == 2:
        x_max = x.max(axis=1, keepdims=True)
        x_exp = np.exp(x - x_max)
        return x_exp / x_exp.sum(axis=1, keepdims=True)


# Exponential Linear Unit (ELU) 
def elu(x) -> np.ndarray:
    x = np.array(x)
    return np.where(x > 0, x, np.exp(x) - 1)

# Gaussian Error Linear Unit (GELU) activation function
def gelu(x) -> np.ndarray:
    x = np.array(x)
    efc_vec = np.vectorize(math.erf)
    return 0.5 * x * (1 + efc_vec(x / math.sqrt(2)))

# Scaled Exponential Linear Unit (SELU) activation function
def selu(x, lam=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717) -> np.ndarray:
    x = np.array(x)
    return lam * np.where(x > 0, x, alpha * (np.exp(x) - 1))

# Swish activation function
def swish(x) -> np.ndarray:
    x = np.array(x)
    return x * sigmoid(x)

# Leaky ReLU implementation
def leaky_relu(x, alpha=0.01) -> np.ndarray:
    x = np.array(x)
    return np.where(x > 0, x, alpha * x)

# Compute the Gaussian Error Linear Unit (exact version using erf)
def gelu_exact(x) -> np.ndarray:
    x = np.array(x)
    efc_vec = np.vectorize(math.erf)
    return 0.5 * x * (1 + efc_vec(x / math.sqrt(2)))

# ReLU activation function
def relu(x) -> np.ndarray:
    x = np.array(x)
    return np.where(x > 0, x, 0)


# Activation Functions Dictionary
ACTIVATION_FUNCTIONS = {
    "RELU": relu,
    "SIGMOID": sigmoid,
    "TANH": tanh,
    "SOFTMAX": softmax,
    "ELU": elu,
    "GELU": gelu,
    "SELU": selu,
    "SWISH": swish,
    "LEAKY_RELU": leaky_relu,
    "GELU_EXACT": gelu_exact
}