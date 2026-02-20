import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """

    A_np = np.array(A)
    m, n = A_np.shape

    A_T = [[0 for _ in range(m)] for _ in range(n)]
    for j in range(n):
        for i in range(m):
            A_T[j][i] = A_np[i][j]

    return np.array(A_T)