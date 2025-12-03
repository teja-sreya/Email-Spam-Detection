import numpy as np
# activation functions
def relu(x):
    return np.maximum(0, x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#  weights
W1 = np.array([0.5, -0.2, 0.3])      # Hidden Layer 1 weights
W2 = np.array([0.4, 0.1, -0.5])      # Hidden Layer 2 weights
Wout = 1                              # Output layer weight

def forward(x):

    # Hidden Layer 1 : ReLU
    z1 = np.sum(x * W1)
    h1 = relu(z1)
    # Hidden Layer 2 : Sigmoid
    z2 = h1 * np.sum(W2)
    h2 = sigmoid(z2)
    # Output : Sigmoid
    z3 = h2 * Wout
    output = sigmoid(z3)
    return output



x = np.array([1, 0, 1])
result = forward(x)
print("Final Output:", result)


#Final Output: 0.6224593312018546
