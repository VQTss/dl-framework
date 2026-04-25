import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    x = np.array(x)
    y = np.array(y)
    if x.shape != y.shape:
        raise ValueError("Vectors must have the same shape")
    result = np.sum(np.power(x - y, 2))
    return np.sqrt(result)
    


x = [1,2,3]
y = [2,2]
print(euclidean_distance(x, y))