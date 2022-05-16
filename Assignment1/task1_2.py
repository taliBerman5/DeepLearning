import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import math


def SGD(Grad, X, Y, w, epochs):
    norms = [LA.norm(Y)]
    lr = 0.1
    batch = 1
    m = len(X)
    for i in range(epochs):
        rnd_ind = np.random.permutation(m)
        if i % 50 == 0:
            lr *= 0.5

        for b in range(math.floor(m / batch)):
            ind = rnd_ind[b * batch: (b + 1) * batch]
            Xb = X[ind]
            Yb = Y[ind]
            curr_grad = Grad(Xb, Yb, w) + 0.001 * w
            w = w - lr * curr_grad

        norms.append(LA.norm(Grad(X, Y, w) + 0.001 * w))

    return w, norms


def LS_grad(X, Y, w):
    m = len(X)
    grad = (X @ w - Y) @ X
    return 1 / m * grad


def plot(norms):
    plt.figure()
    plt.semilogy(norms, label="Grad Norm")
    plt.title("Least Squares with SGD")
    plt.ylabel("grad norm")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()

def LS_test():
    X = np.asarray([[3, -1, 1], [1, -1, 2], [-1, -3, 5], [1, 3, 4]])
    Y = np.asarray([13, 20, 0, 14])
    w = np.zeros(3)

    w, norms = SGD(LS_grad, X, Y, w, epochs=1000)
    plot(norms)
    print(f'SGD result: {w}')
    print(f'Exact result: {np.linalg.inv(X.T @ X) @ X.T @ Y}')

# LS_test()

