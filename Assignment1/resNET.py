import math
import numpy.linalg as LA
from Assignment1.calculations import *
from Assignment1.data import *


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

    linear_cache = (A_prev, W1, W2, b)
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
    success_training = [check_success(Xt, Yt, parameters, classify_resnet)]
    success_validation = [check_success(Xv, Yv, parameters, classify_resnet)]
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
        success_training.append(check_success(Xt, Yt, parameters, classify_resnet))
        success_validation.append(check_success(Xv, Yv, parameters, classify_resnet))

    return parameters, success_training, success_validation


def classify_resnet(X, parameters):
    m = len(X[0])
    L = len(parameters) // 3
    l = len(parameters["b" + str(L)])
    AL, caches = forward_pass(X, parameters, np.tanh)
    labels = np.argmax(AL, axis=0)
    clasify_matrix = np.zeros((l, m))
    clasify_matrix[labels, np.arange(m)] = 1
    return clasify_matrix


def update_parameters_grad_test(parameters, parameters_D, eps):
    L = len(parameters) // 3
    new_parameters = {}
    for l in range(1, L + 1):
        new_parameters["W1_" + str(l)] = parameters["W1_" + str(l)] + eps * parameters_D["W1_" + str(l)]
        new_parameters["W2_" + str(l)] = parameters["W2_" + str(l)] + eps * parameters_D["W2_" + str(l)]
        new_parameters["b" + str(l)] = parameters["b" + str(l)] + eps * parameters_D["b" + str(l)]
    return new_parameters


def jacobian_test_WB_resnet():
    X = np.random.randn(3, 1)
    W1 = np.random.randn(3, 3)
    W2 = np.random.randn(3, 3)
    b = np.random.randn(3, 1)

    v_u = np.random.randn(3, 1)
    u = v_u / np.linalg.norm(v_u)

    D_W1 = np.random.rand(3, 3)
    D_W1 = (1 / LA.norm(D_W1)) * D_W1

    D_W2 = np.random.rand(3, 3)
    D_W2 = (1 / LA.norm(D_W2)) * D_W2

    D_b = np.random.rand(3, 1)
    D_b = (1 / LA.norm(D_b)) * D_b

    zero_loss = []
    one_loss = []
    epsilon = 1
    f0 = g(X + W2 @ np.tanh(W1 @ X + b), u).flatten()
    dX, dW1, dW2, db = calc_grad_res(X, W1, W2, b, u)

    for i in range(20):
        w1_eps_k = W1 + epsilon * D_W1
        w2_eps_k = W2 + epsilon * D_W2
        b_eps_k = b + epsilon * D_b
        fk = g(X + w2_eps_k @ np.tanh(w1_eps_k @ X + b_eps_k), u).flatten()
        f1 = f0 + epsilon * (np.ndarray.flatten(D_W1) @ np.ndarray.flatten(dW1)) + epsilon * (
                np.ndarray.flatten(D_W2) @ np.ndarray.flatten(dW2)) + epsilon * (
                     np.ndarray.flatten(D_b) @ np.ndarray.flatten(db))
        zero_loss.append(abs(fk - f0))
        one_loss.append(abs(fk - f1))
        epsilon *= 0.5

    plot_grad_test(zero_loss, one_loss, 'resNet \nJacobian test by W and b')


def jacobian_test_X_resnet():
    X = Yt_Peaks[:, 0].reshape(2, 1)
    W1 = np.random.rand(2, 2)
    W2 = np.random.rand(2, 2)
    b = np.random.rand(2, 1)

    v_u = np.random.randn(2, 1)
    u = v_u / np.linalg.norm(v_u)

    D_X = np.random.rand(2, 1)
    D_X = (1 / LA.norm(D_X)) * D_X

    zero_loss = []
    one_loss = []
    epsilon = 1
    f0 = g(X + W2 @ np.tanh(W1 @ X + b), u).flatten()
    dX, dW1, dW2, db = calc_grad_res(X, W1, W2, b, u)
    for i in range(20):
        x_eps_k = X + epsilon * D_X
        fk = g(x_eps_k + W2 @ np.tanh(W1 @ x_eps_k + b), u).flatten()
        f1 = f0 + epsilon * (np.ndarray.flatten(D_X) @ np.ndarray.flatten(dX))
        zero_loss.append(abs(fk - f0))
        one_loss.append(abs(fk - f1))
        epsilon *= 0.5
    plot_grad_test(zero_loss, one_loss, 'resNet\nJacobian test by X')


def stack_w_b_grads(grads):
    stack = []
    l = len(grads) // 4
    stack.append(grads["dW1_1"].flatten())
    stack.append(grads["db1"].flatten())

    for i in range(2, l):
        stack.append(grads["dW1_" + str(i)].flatten())
        stack.append(grads["dW2_" + str(i)].flatten())
        stack.append(grads["db" + str(i)].flatten())

    stack.append(grads["dW1_" + str(l)].flatten())
    stack.append(grads["db" + str(l)].flatten())
    return np.concatenate(stack, axis=0)


def stack_parametersD(parameters_D):
    stack = []
    l = len(parameters_D) // 3

    stack.append(parameters_D["W1_1"].flatten())
    stack.append(parameters_D["b1"].flatten())

    for i in range(2, l):
        stack.append(parameters_D["W1_" + str(i)].flatten())
        stack.append(parameters_D["W2_" + str(i)].flatten())
        stack.append(parameters_D["b" + str(i)].flatten())

    stack.append(parameters_D["W1_" + str(l)].flatten())
    stack.append(parameters_D["b" + str(l)].flatten())
    return np.concatenate(stack, axis=0)


def grad_Test_resnet():
    X = Yt_Peaks
    Y = Ct_Peaks
    eps = 0.1
    linearly_grad_test = []
    quadratically_grad_test = []

    parameters = init_params([2, 10, 10, 5], D=0)
    parameters_D = init_params([2, 10, 10, 5], D=1)
    num_params = len(parameters) // 3

    AL, caches = forward_pass(X, parameters, np.tanh)
    grads = backpropagation(AL, Y, caches)

    f0 = softmax_regression(caches[-1][0][0], parameters["W1_" + str(num_params)], parameters["b" + str(num_params)], Y)
    stack_grads = stack_w_b_grads(grads)
    stack_parameters_D = stack_parametersD(parameters_D)

    for i in range(20):
        new_parameters = update_parameters_grad_test(parameters, parameters_D, eps)
        AL, caches = forward_pass(X, new_parameters, np.tanh)
        fK = softmax_regression(caches[-1][0][0], new_parameters["W1_" + str(num_params)],
                                new_parameters["b" + str(num_params)],
                                Y)

        f1 = f0 + eps * stack_grads @ stack_parameters_D

        linearly_grad_test.append(abs(fK - f0))
        quadratically_grad_test.append(abs(fK - f1))

        eps *= 0.5

    plot_grad_test(linearly_grad_test, quadratically_grad_test, "resNet\nGrad test for the whole Neural Network")


def test_data(Xt, Ct, Xv, Cv, hidden_layer, type, lr, batch):
    epochs = 100
    n = len(Xt)
    l = len(Ct)
    layer_dims = [n] + hidden_layer + [l]

    parameters, success_training, success_validation = SGD_resnet(Xt, Ct, Xv, Cv, layer_dims, epochs, batch, lr)
    plot(success_training, success_validation, type, lr, batch, title="resNet using SGD")


def test_data_200(Xt, Ct, Xv, Cv, hidden_layer, type, lr, batch):
    epochs = 200
    n = len(Xt)
    l = len(Ct)
    layer_dims = [n] + hidden_layer + [l]
    X_200, C_200 = sample(Xt, Ct, 200)

    parameters, success_training, success_validation = SGD_resnet(X_200, C_200, Xv, Cv, layer_dims, epochs, batch, lr)
    plot(success_training, success_validation, type, lr, batch, title="resNet using SGD - 200 training samples")


# jacobian_test_WB_resnet()
# jacobian_test_X_resnet()
# grad_Test_resnet()
#
# test_data(Yt_SwissRoll, Ct_SwissRoll, Yv_SwissRoll, Cv_SwissRoll, [10,10,10], "Swiss Roll", lr=0.5, batch=1000)
# test_data_200(Yt_SwissRoll, Ct_SwissRoll, Yv_SwissRoll, Cv_SwissRoll, [10, 10, 10], "Swiss Roll", lr=0.5, batch=100)
#
# test_data(Yt_Peaks, Ct_Peaks, Yv_Peaks, Cv_Peaks, [10, 10, 10], "Peaks", lr=0.5, batch=100)
# test_data_200(Yt_Peaks, Ct_Peaks, Yv_Peaks, Cv_Peaks, [10, 10, 10], "Peaks", lr=0.1, batch=100)
#
# test_data(Yt_GMM, Ct_GMM, Yv_GMM, Cv_GMM, [10,10,10], "GMM", lr=0.5, batch=100)
# test_data_200(Yt_GMM, Ct_GMM, Yv_GMM, Cv_GMM, [10, 10, 10], "GMM", lr=0.5, batch=100)
