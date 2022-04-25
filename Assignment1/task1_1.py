import numpy as np
import scipy.io as sio
import numpy.linalg as LA
import matplotlib.pyplot as plt

GMM = sio.loadmat('GMMData.mat')
Peaks = sio.loadmat('PeaksData.mat')
SwissRoll = sio.loadmat('SwissRollData.mat')

Ct = SwissRoll["Ct"]
Cv = SwissRoll["Cv"]
Yt = SwissRoll["Yt"]
Yv = SwissRoll["Yv"]


def eta_calc(X, W):
    return np.max(X.T @ W, axis=1)


def softmax(X, W):
    eta = eta_calc(X, W)
    return np.exp((X.T @ W).T - eta) / np.sum(np.exp((X.T @ W).T - eta).T,
                                              axis=1)


def softmax_regression(X, C, W):
    n, m = np.shape(X)
    return -1 / m * np.sum(C * np.log(softmax(X, W)).T)


def softmax_regression_grad(X, C, W):
    n, m = np.shape(X)
    gradf = X @ (softmax(X, W) - C.T).T
    return 1 / m * gradf


def grad_test():
    X = Yt
    C = Ct.T
    X = np.vstack((X, np.ones(len(X[0]))))  # added dimension for the bias
    epsilon = 0.1
    n = len(X)
    l = len(C[1])
    W = np.random.rand(n, l)
    D = np.random.rand(n, l)
    D = (1 / LA.norm(D)) * D  # normelize to 1
    F0 = softmax_regression(X, C, W)
    g0 = softmax_regression_grad(X, C, W)
    linearly_grad_test = []
    quadratically_grad_test = []
    for k in range(20):
        Fk = softmax_regression(X, C, W + epsilon * D)
        F1 = F0 + epsilon * (np.ndarray.flatten(g0, 'F') @ np.ndarray.flatten(D, 'F'))
        linearly_grad_test.append(abs(Fk - F0))
        quadratically_grad_test.append(abs(Fk - F1))
        epsilon = epsilon * 0.5

    axis = [i for i in range(20)]
    plt.figure()
    plt.semilogy(axis, linearly_grad_test, label="Zero order approx")
    plt.semilogy(axis, quadratically_grad_test, label="First order approx")
    plt.xlabel("k")
    plt.ylabel("error")
    plt.title("Grad test for softmax regression")
    plt.legend()
    plt.show()


# grad_test()
