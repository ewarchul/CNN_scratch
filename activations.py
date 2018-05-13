import numpy as np

def relu(x):
    """Funkcja aktywacji ReLU"""
    return (x+abs(x))/2


def relu_b(error, inUnits):
    """Funkcja realizacuja propagacje wsteczna dla funkcji aktywacji ReLU (jej pochodna)"""
    def ReLU(y):
		return 1 if y > 0 else 0
    ReLU = np.vectorize(ReLU)
    return error*ReLU(inUnits)



