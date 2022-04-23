import numpy as np
import scipy.io as sio
import numpy.linalg as LA
import matplotlib.pyplot as plt

GMM = sio.loadmat('GMMData.mat')
Peaks = sio.loadmat('PeaksData.mat')
SwissRoll = sio.loadmat('SwissRollData.mat')

Ct = GMM["Ct"]
Cv = GMM["Cv"]
Yt = GMM["Yt"]
Yv = GMM["Yv"]


def eta_calc(X, W):
    return np.max(X.T @ W, axis=1)


def softmax(X, W):
    eta = eta_calc(X, W)
    return np.exp((X.T @ W).T - eta) / np.sum(np.exp((X.T @ W).T - eta).T,
                                              axis=1)  # mabye needs transpose after subtruct eta


def softmax_regression(C, X, W):
    n, m = np.shape(X)
    return -1 / m * np.sum(C * np.log(softmax(X, W)))


def softmax_regression_grad(C, X, W):
    n, m = np.shape(X)
    gradf = X @ (softmax(X, W) - C)  # TODO: need transpose?
    return 1 / m * gradf


def grad_test():
    X = Yt
    C = Ct
    X = np.vstack((X, np.ones(len(X[0]))))  # added dimension for the bias
    epsilon = 0.1
    n = len(X)
    l = len(C[1])
    W = np.random.rand(n, l)
    D = np.random.rand(n, len(X[0]))
    D = (1 / LA.norm(D)) * D  # normelize to 1
    F0 = softmax_regression(C, X, W)
    g0 = softmax_regression_grad(C, X, W)
    linearly_grad_test = []
    quadratically_grad_test = []
    for k in range(1, 20):
        Fk = softmax_regression(C, X + epsilon * D, W)
        F1 = F0 + epsilon * (g0 @ D)
        linearly_grad_test.append(abs(Fk - F0))
        quadratically_grad_test.append(abs(Fk - F1))
        epsilon = epsilon * 0.5

    axis = [i for i in range(20)]
    plt.figure()
    plt.semilogy(axis, linearly_grad_test, lable="Zero order approx")
    plt.semilogy(axis, quadratically_grad_test, lable="First order approx")
    plt.xlabel("k")
    plt.ylabel("error")
    plt.title("Grad test for softmax regression")
    plt.legend()
    plt.show()


grad_test()
# X = Yt
# C = Ct
# X = np.vstack((X, np.ones(len(X[0]))))  # added dimension for the bias
#
# a = np.asarray([[1, 2, 3], [10, 11, 12]])
# print(a)
# print()
# print((a.T - np.asarray([1, 2])).T)
