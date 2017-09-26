import numpy as np

def relu(z):
    return (1.0/(1.0+np.exp(-z)))*(1.0-(1.0/(1.0+np.exp(-z))))


x = np.array([-3, -.5, -.2, 0, .2, .5, 3])
print(x)
print(relu(x))
