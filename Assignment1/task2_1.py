import numpy as np
import math
import scipy.io as sio
import numpy.linalg as LA
import matplotlib.pyplot as plt
from Assignment1.task1_3 import check_success, Ct_SwissRoll, Cv_SwissRoll, Yt_SwissRoll, Yv_SwissRoll


def tanh_grad(Z):
    return np.ones(np.shape(Z)) - np.power(np.tanh(Z), 2)




def eta_calc(Z):
    return np.max(Z, axis=1)


def softmax_regression(A, W, b, Y):
    Z = A.T @ W + b
    n, m = np.shape(A)
    return -1 / m * np.sum(Z * np.log(softmax(Z)).T)

def softmax(Z):
    eta = eta_calc(Z)
    return np.exp(Z.T - eta) / np.sum(np.exp(Z.T - eta).T, axis=1).T


def softmax_regression_grad_by_A(A, W, b, Y):
    n, m = np.shape(A)
    Z = A.T @ W + b
    gradf = W.T @ (softmax(Z).T - Y).T
    return (1 / m) * gradf

def softmax_regression_grad(A, W, b, Y):
    n, m = np.shape(A)
    Z = A.T @ W + b
    gradf = A @ (softmax(Z) - Y.T).T
    return 1 / m * gradf

def init_params(layers_dims):
    parameters = {}
    L = len(layers_dims)  # number of layers

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def calc_layer(A_prev, W, b, activation_func):
    Z = W @ A_prev + b
    A = activation_func(Z)

    linear_cache = (A_prev, W, b)
    activation_cache = Z

    return A, (linear_cache, activation_cache)


def forward_pass(X, parameters, activation_func):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers
    for l in range(1, L):
        A_prev = A
        A, cache = calc_layer(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation_func)
        caches.append(cache)

    WL = parameters['W' + str(L)]
    bL =  parameters['b' + str(L)]
    ZL = WL @ A + bL
    AL = softmax(ZL)
    linear_cache = (A, WL, bL)
    caches.append((linear_cache, ZL))

    return AL, caches


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ.T, A_prev) / m
    db = np.squeeze(np.sum(dZ, axis=1, keepdims=True)) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db



def backpropagation(AL, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    A_prev, W, b = caches[-1][0]
    ZL = caches[-1][1]
    dAL = softmax_regression_grad_by_A(ZL, W, b, Y)

    current_cache = caches[-1]
    softmax_regression_grad(dAL, Y, current_cache[0][1].T)
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(dAL, current_cache[0])


    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dAl = grads["dA" + str(l + 2)]
        grads["dA" + str(l + 1)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = linear_backward(tanh_grad(current_cache[0][1] @ dAl + current_cache[0][2]), current_cache[0])  #TODO: check tanh_grad

    return grads



def update_parameters(parameters, grads, lr):
    L = len(parameters) // 2  # number of layers

    for l in range(1, L+1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - lr * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - lr * grads["db" + str(l)]

    return parameters



def SGD_nn(Xt, Yt, Xv, Yv, layers_dims, epochs, batch, lr):
    parameters = init_params(layers_dims)
   # success_training = [check_success(Xt, Yt, W)]
   # success_validation = [check_success(Xv, Yv, W)]
    costs = []
    m = len(Xt[0])
    for i in range(epochs):
        rnd_ind = np.random.permutation(m)
        if i % 50 == 0:
            lr *= 0.5

        for b in range(math.floor(m / batch)):
            ind = rnd_ind[b * batch: (b + 1) * batch]
            X_ind = Xt[:, ind]
            Y_ind = Yt[:, ind]

            AL, caches = forward_pass(X_ind, parameters, np.tanh)
            costs.append(softmax_regression(AL.T, Y_ind))  #todo:change
            grads = backpropagation(AL, Y_ind, caches)
            parameters = update_parameters(parameters, grads, lr)


   #     success_training.append(check_success(Xt, Yt, W))
    #    success_validation.append(check_success(Xv, Yv, W))

    return parameters, costs


def test_data(Xt, Ct, Xv, Cv, type, lr, batch):
    epochs = 100
    n = len(Xt)
    l = len(Ct)
    layer_dims = [n, 4, 5, 3, l]

    parameters, costs = SGD_nn(Xt, Ct, Xv, Cv, layer_dims, epochs, batch, lr)

test_data(Yt_SwissRoll, Ct_SwissRoll, Yv_SwissRoll, Cv_SwissRoll, "Swiss Roll", 0.1, 1000)
# def clasify(X, parametrs):
#     l = len(W[1])
#     m = len(X[1])
#     prob = softmax(X, W)
#     labels = np.argmax(prob, axis=0)
#     clasify_matrix = np.zeros((l, m))
#     clasify_matrix[labels, np.arange(m)] = 1
#     return clasify_matrix


# def check_success(X, C, W):
#     m = len(X[1])
#     clasify_matrix = clasify(X, W)
#     success = np.sum(1 - np.abs(clasify_matrix - C), axis=1)[0]
#     return success / m



