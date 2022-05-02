import numpy as np
import math
import numpy.linalg as LA
import matplotlib.pyplot as plt
from Assignment1.task1_3 import Ct_SwissRoll, Cv_SwissRoll, Yt_SwissRoll, Yv_SwissRoll, Yt_Peaks, Ct_Peaks, plot
from Assignment1.task1_1 import plot_grad_test
from Assignment1.Neural_Network import softmax, calc_layer, softmax_regression_grad, softmax_regression, calc_tanh_grad


def init_params(layers_dims, D):
    parameters = {}
    L = len(layers_dims)  # number of layers

    for l in range(1, L):
        parameters['W1_' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1])
        parameters['W2_' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1])
        parameters['b' + str(l)] = np.random.randn(layers_dims[l], 1)
        if D == 1:  # norm 1 for grad test
            parameters['W1_' + str(l)] = (1 / LA.norm(parameters['W1_' + str(l)])) * parameters['W1_' + str(l)]
            parameters['W2_' + str(l)] = (1 / LA.norm(parameters['W2_' + str(l)])) * parameters['W2_' + str(l)]
            parameters['b' + str(l)] = (1 / LA.norm(parameters['b' + str(l)])) * parameters['b' + str(l)]

    return parameters


def calc_res_layer(A_prev, W1, W2, b, activation_func):
    Z = W1 @ A_prev + b
    A = activation_func(Z)
    res_A = A_prev + W2 @ A

    linear_cache = (A_prev, W1, W2, b)  # TODO: decide what to return
    activation_cache = Z

    return res_A, (linear_cache, activation_cache)


def forward_pass(X, parameters, activation_func):
    caches = []
    A = X
    L = len(parameters) // 3  # number of layers

    A_prev = A
    A, cache = calc_layer(A_prev, parameters['W1_1'], parameters['b1'], activation_func)
    caches.append(cache)

    for l in range(2, L):
        A_prev = A
        A, cache = calc_res_layer(A_prev, parameters['W1_' + str(l)], parameters['W2_' + str(l)],
                                  parameters['b' + str(l)], activation_func)
        caches.append(cache)

    WL = parameters['W1_' + str(L)]
    bL = parameters['b' + str(L)]
    ZL = WL @ A + bL
    AL = softmax(ZL)
    linear_cache = (A, WL, -1, bL)
    caches.append((linear_cache, ZL))

    return AL, caches


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = dZ @ A_prev.T
    db = np.reshape(np.squeeze(np.sum(dZ, axis=1, keepdims=True)), b.shape)
    dA_prev = W.T @ dZ

    return dA_prev, dW, -1, db


def calc_grad_res(A, W1, W2, b, v):
    Z = W1 @ A + b
    tanh_grad = calc_tanh_grad(Z)
    W2_grad_tanh = tanh_grad * (W2.T @ v)
    db = np.reshape(np.squeeze(np.sum(W2_grad_tanh, axis=1, keepdims=True)), b.shape)
    dW1 = W2_grad_tanh @ A.T
    dW2 = v @ np.tanh(Z).T
    dA = v + W1.T @ W2_grad_tanh

    return dA, dW1, dW2, db


def backpropagation(AL, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    A_prev, W1, W2, b = caches[-1][0]
    dAL, dWL, dbL = softmax_regression_grad(A_prev, W1, b, Y)
    grads["dA" + str(L)], grads["dW1_" + str(L)], grads["dW2_" + str(L)], grads["db" + str(L)] = dAL, dWL, -1, dbL

    for l in reversed(range(1, L - 1)):
        A_prev, W1, W2, b = caches[l][0]
        dAl = grads["dA" + str(l + 2)]
        grads["dA" + str(l + 1)], grads["dW1_" + str(l + 1)], grads["dW2_" + str(l + 1)], grads[
            "db" + str(l + 1)] = calc_grad_res(A_prev, W1, W2, b, dAl)

    current_cache = caches[0]
    A_prev, W1, b = current_cache[0]
    dAl = grads["dA2"]
    grads["dA1"], grads["dW1_1"], grads["dW2_1"], grads["db1"] = linear_backward(
        (calc_tanh_grad(W1 @ A_prev + b)) * dAl, current_cache[0])

    return grads


def update_parameters(parameters, grads, lr):
    L = len(parameters) // 3  # number of layers

    for l in range(1, L + 1):
        parameters["W1_" + str(l)] = parameters["W1_" + str(l)] - lr * grads["dW1_" + str(l)]
        parameters["W2_" + str(l)] = parameters["W2_" + str(l)] - lr * grads["dW2_" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - lr * grads["db" + str(l)]

    return parameters


def SGD_resnet(Xt, Yt, Xv, Yv, layers_dims, epochs, batch, lr):
    parameters = init_params(layers_dims, D=0)
    success_training = [check_success(Xt, Yt, parameters)]
    success_validation = [check_success(Xv, Yv, parameters)]
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
            grads = backpropagation(AL, Y_ind, caches)
            parameters = update_parameters(parameters, grads, lr)

        AL, caches = forward_pass(Xt, parameters, np.tanh)
        A_prev, W1, W2, b = caches[-1][0]
        costs.append(softmax_regression(A_prev, W1, b, Yt))
        success_training.append(check_success(Xt, Yt, parameters))
        success_validation.append(check_success(Xv, Yv, parameters))

    return parameters, success_training, success_validation


def clasify(X, parameters):
    m = len(X[1])
    L = len(parameters) // 3
    l = len(parameters["b" + str(L)])
    AL, caches = forward_pass(X, parameters, np.tanh)
    labels = np.argmax(AL, axis=0)
    clasify_matrix = np.zeros((l, m))
    clasify_matrix[labels, np.arange(m)] = 1
    return clasify_matrix


def check_success(X, C, parameters):
    m = len(X[1])
    clasify_matrix = clasify(X, parameters)
    success = np.sum(1 - np.abs(clasify_matrix - C), axis=1)[0]
    return success / m

def update_parameters_grad_test(parameters, parameters_D, eps):
    L = len(parameters) // 3
    new_parameters = {}
    for l in range(1, L + 1):
        new_parameters["W1_" + str(l)] = parameters["W1_" + str(l)] + eps * parameters_D["W1_" + str(l)]
        new_parameters["W2_" + str(l)] = parameters["W2_" + str(l)] + eps * parameters_D["W2_" + str(l)]
        new_parameters["b" + str(l)] = parameters["b" + str(l)] + eps * parameters_D["b" + str(l)]
    return new_parameters




def test_data(Xt, Ct, Xv, Cv, type, lr, batch):
    epochs = 100
    n = len(Xt)
    l = len(Ct)
    layer_dims = [n, 10, 10, 10, l]

    parameters, success_training, success_validation = SGD_resnet(Xt, Ct, Xv, Cv, layer_dims, epochs, batch, lr)
    plot(success_training, success_validation, type, lr, batch, title="resNet using SGD")


test_data(Yt_SwissRoll, Ct_SwissRoll, Yv_SwissRoll, Cv_SwissRoll, "Swiss Roll", lr=0.5, batch=100)