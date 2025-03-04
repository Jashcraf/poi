import numpy as np

def broadcast_transpose(M):
    return np.einsum('...ij->...ji', M)