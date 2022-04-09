import numpy as np


def leastSquares(data, label):
    # Sum of squared error should be minimized
    #
    # INPUT:
    # data        : Training inputs  (num_samples x dim)
    # label       : Training targets (num_samples x 1)
    #
    # OUTPUT:
    # weights     : weights   (dim x 1)
    # bias        : bias term (scalar)

    #####Insert your code here for subtask 1a#####
    data = np.insert(data, 0, values= 1, axis=1)
    inv = np.linalg.pinv(data)
    w_new = np.matmul(inv, label)
    bias = w_new[0]
    weight = np.delete(w_new, [0])

    # Extend each datapoint x as [1, x]
    # (Trick to avoid modeling the bias term explicitly)
    return weight, bias
