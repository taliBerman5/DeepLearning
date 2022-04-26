import numpy as np
import math
import scipy.io as sio
import numpy.linalg as LA
import matplotlib.pyplot as plt
from Assignment1.task1_3 import check_success


def tanh_grad(Z):
    return np.ones(np.shape(Z)) - np.power(np.tanh(Z), 2)


def softmax_grad(A, W):   #TODO: ask Eran
    4


def eta_calc(A):
    return np.max(A, axis=1)


def softmax_regression(A, Y):
    n, m = np.shape(A)
    return -1 / m * np.sum(Y * np.log(softmax(A)).T)

def softmax(A):
    eta = eta_calc(A)
    return np.exp(A.T - eta) / np.sum(np.exp(A.T - eta).T, axis=1)


def softmax_regression_grad_by_A(A, Y, W):
    n, m = np.shape(A)
    gradf = W @ (softmax(A, W) - Y.T).T
    return 1 / m * gradf


def init_params(layers_dims):
    parameters = {}
    L = len(layers_dims)  # number of layers

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros(layers_dims[l], 1)

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
    for l in range(1, L - 1):
        A_prev = A
        A, cache = calc_layer(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation_func)
        caches.append(cache)

    WL = parameters['W' + str(L - 1)]
    bL =  parameters['b' + str(L - 1)]
    ZL = WL @ A + bL                        #TODO: ask Eran how should we calc the last layer
    AL = softmax(ZL)
    linear_cache = (A, WL, bL)
    caches.append((linear_cache, ZL))

    return AL, caches


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, cache[0].T) / m
    db = np.squeeze(np.sum(dZ, axis=1, keepdims=True)) / m
    dA_prev = np.dot(cache[1].T, dZ)

    return dA_prev, dW, db



def backpropagation(AL, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    dAL = softmax_regression_grad_by_A(AL, Y, caches[-1][1])  #TODO: check place W in caches

    current_cache = caches[-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(softmax_grad(dAL, current_cache[1]), current_cache[0])


    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dAl = grads["dA" + str(l + 2)]
        grads["dA" + str(l + 1)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = linear_backward(tanh_grad(current_cache[1] @ dAl + current_cache[2]), current_cache[0])  #TODO: check tanh_grad

    return grads



def update_parameters(parameters, grads, lr):
    L = len(parameters) // 2  # number of layers

    for l in range(1, L+1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - lr * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - lr * grads["db" + str(l)]

    return parameters



def SGD_nn(Grad, Xt, Yt, Xv, Yv, layers_dims, epochs, batch, lr):
    parameters = init_params(layers_dims)
   # success_training = [check_success(Xt, Yt, W)]
   # success_validation = [check_success(Xv, Yv, W)]
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
            cost = softmax_regression(AL, Y_ind)
            grads = backpropagation(AL, Y_ind, caches)
            parameters = update_parameters(parameters, grads, lr)


   #     success_training.append(check_success(Xt, Yt, W))
    #    success_validation.append(check_success(Xv, Yv, W))

    return W, success_training, success_validation




def clasify(X, parametrs):
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
