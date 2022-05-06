import numpy as np
import matplotlib.pyplot as plt

def g(f, v):
    return f.T @ v


def calc_tanh_grad(Z):
    return np.ones(np.shape(Z)) - np.power(np.tanh(Z), 2)

def softmax(Z):
    eta = np.max(Z)
    return np.exp(Z - eta) / np.sum(np.exp(Z - eta), axis=0)

def softmax_regression(A, W, b, Y):
    Z = W @ A + b
    n, m = np.shape(A)
    return - np.sum(Y * np.log(softmax(Z))) / m

def softmax_regression_grad(A, W, b, Y):
    n, m = np.shape(A)
    Z = W @ A + b
    soft_max_minus_Y = (softmax(Z) - Y)
    gradA = W.T @ soft_max_minus_Y / m
    gradW = (soft_max_minus_Y @ A.T) / m
    gradb = (soft_max_minus_Y / m).sum(axis=1, keepdims=True)
    return gradA, gradW, gradb


def calc_layer(A_prev, W, b, activation_func):
    Z = W @ A_prev + b
    A = activation_func(Z)

    linear_cache = (A_prev, W, b)
    activation_cache = Z

    return A, (linear_cache, activation_cache)

def sample(X, Y, amount):
    rnd_ind = np.random.permutation(X.shape[1])
    ind = rnd_ind[: amount]
    X_ind = X[:, ind]
    Y_ind = Y[:, ind]

    return X_ind, Y_ind


def plot_grad_test(zero_order, first_order, title):
    axis = [i for i in range(20)]
    plt.figure()
    plt.semilogy(axis, zero_order, label="Zero order approx")
    plt.semilogy(axis, first_order, label="First order approx")
    plt.xlabel("k")
    plt.ylabel("error")
    plt.title(title)
    plt.legend()
    plt.show()


def plot(success_percentage_train, success_percentage_validation, type, lr, batch, title):
    plt.figure()
    plt.plot([i for i in range(len(success_percentage_train))], success_percentage_train,
             label="Success percentage train")
    plt.plot([i for i in range(len(success_percentage_validation))], success_percentage_validation,
             label="Success percentage validation")
    plt.xlabel("epochs")
    plt.ylabel("Success rate")
    plt.title(f'{title} - {type} \n lr = {lr}, batch = {batch}')
    plt.legend()
    plt.show()
