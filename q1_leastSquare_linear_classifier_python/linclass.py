import numpy as np

def linclass(weight, bias, data):
    # Linear Classifier
    #
    # INPUT:
    # weight      : weights                (dim x 1)
    # bias        : bias term              (scalar)
    # data        : Input to be classified (num_samples x dim)
    #
    # OUTPUT:
    # class_pred       : Predicted class (+-1) values  (num_samples x 1)

    #####Insert your code here for subtask 1b#####
    y1 = np.matmul(data, np.transpose(weight))
    print(y1)
    y = y1 + bias
    class_pred = ((y>=0)*2)-1

    # Perform linear classification i.e. class prediction
    return class_pred


