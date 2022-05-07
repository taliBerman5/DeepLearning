import numpy as np
import scipy.io as sio
import numpy.linalg as LA
from Assignment1.calculations import plot_grad_test

GMM = sio.loadmat('GMMData.mat')
Ct = GMM["Ct"]
Cv = GMM["Cv"]
Yt = GMM["Yt"]
Yv = GMM["Yv"]


def eta_calc(X, W):
    return np.max(X.T @ W, axis=1)


def softmax(X, W):
    eta = eta_calc(X, W)
    return np.exp((X.T @ W).T - eta) / np.sum(np.exp((X.T @ W).T - eta).T, axis=1)


def softmax_regression(X, C, W):
    n, m = np.shape(X)
    return - np.sum(C * np.log(softmax(X, W))) / m


def softmax_regression_grad(X, C, W):
    n, m = np.shape(X)
    gradf = X @ (softmax(X, W) - C).T
    return 1 / m * gradf


def grad_test():
    X = Yt
    C = Ct
    X = np.vstack((X, np.ones(len(X[0]))))  # added dimension for the bias
    epsilon = 0.1

    n = len(X)
    l = len(C)

    W = np.random.rand(n, l)

    D = np.random.rand(n, l)
    D = (1 / LA.norm(D)) * D  # normelize to 1

    F0 = softmax_regression(X, C, W)
    g0 = softmax_regression_grad(X, C, W)
    zero_loss = []
    one_loss = []
    for k in range(20):
        Fk = softmax_regression(X, C, W + epsilon * D)
        F1 = F0 + epsilon * (np.ndarray.flatten(g0) @ np.ndarray.flatten(D))
        zero_loss.append(abs(Fk - F0))
        one_loss.append(abs(Fk - F1))
        epsilon = epsilon * 0.5

    plot_grad_test(zero_loss, one_loss, "Grad test for softmax regression")




grad_test()


