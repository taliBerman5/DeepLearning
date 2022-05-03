import numpy as np
import math
from Assignment1.task1_1 import softmax, softmax_regression_grad
from Assignment1.calculations import plot
from Assignment1.data import *



def SGD(Grad, Xt, Ct, Xv, Cv, W, epochs, batch, lr):
    success_training = [check_success(Xt, Ct, W)]
    success_validation = [check_success(Xv, Cv, W)]
    m = len(Xt[0])
    for i in range(epochs):
        rnd_ind = np.random.permutation(m)
        if i % 50 == 0:
            lr *= 0.5

        for b in range(math.floor(m / batch)):
            ind = rnd_ind[b * batch: (b + 1) * batch]
            X_ind = Xt[:, ind]
            C_ind = Ct[:, ind]
            curr_grad = Grad(X_ind, C_ind.T, W) + 0.001 * W
            W = W - lr * curr_grad

        success_training.append(check_success(Xt, Ct, W))
        success_validation.append(check_success(Xv, Cv, W))

    return W, success_training, success_validation


def clasify(X, W):
    l = len(W[1])
    m = len(X[1])
    prob = softmax(X, W)
    labels = np.argmax(prob, axis=0)
    clasify_matrix = np.zeros((l, m))
    clasify_matrix[labels, np.arange(m)] = 1
    return clasify_matrix


def check_success(X, C, W):
    m = len(X[1])
    clasify_matrix = clasify(X, W)
    success = np.sum(1 - np.abs(clasify_matrix - C), axis=1)[0]
    return success / m


def test_data(Xt, Ct, Xv, Cv, type, lr, batch):
    epochs = 100
    Xt = np.vstack([Xt, np.ones(len(Xt[0]))])
    Xv = np.vstack([Xv, np.ones(len(Xv[0]))])
    n = len(Xt)
    l = len(Ct)
    W = np.random.rand(n, l)
    W, success_percentage_train, success_percentage_validation = SGD(softmax_regression_grad, Xt, Ct, Xv, Cv, W,
                                                                     epochs, batch, lr)
    plot(success_percentage_train, success_percentage_validation, type, lr, batch, title="Softmax minimization using SGD")




test_data(Yt_SwissRoll, Ct_SwissRoll, Yv_SwissRoll, Cv_SwissRoll, "Swiss Roll", lr=0.1, batch=500)
test_data(Yt_GMM, Ct_GMM, Yv_GMM, Cv_GMM, "GMM", lr=0.1, batch=200)
test_data(Yt_Peaks, Ct_Peaks, Yv_Peaks, Cv_Peaks, "Peaks", lr=0.1, batch=200)