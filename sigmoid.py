import numpy as np

        # Função sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada da função sigmoide
def sigmoid_derivada(x):
    return x * (1 - x)
