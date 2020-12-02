import numpy as np


# for forward Propagate  
def sigmoid(x):
    return 1/(1+np.exp(-x))





# for back Propagate 
def sigmoid_drivative(x):
    return x * (x -1)