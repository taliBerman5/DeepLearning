import numpy.linalg as LA
import math
from Assignment1.calculations import *
from Assignment1.data import *

def init_params(layers_dims, D):
    parameters = {}
    L = len(layers_dims)  # number of layers

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1])
        parameters['b' + str(l)] = np.random.randn(layers_dims[l], 1)
        if D == 1:  # norm 1 for grad test
            parameters['W' + str(l)] = (1 / LA.norm(parameters['W' + str(l)])) * parameters['W' + str(l)]
            parameters['b' + str(l)] = (1 / LA.norm(parameters['b' + str(l)])) * parameters['b' + str(l)]

    return parameters



def forward_pass(X, parameters, activation_func):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers
    for l in range(1, L):
        A_prev = A
        A, cache = calc_layer(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation_func)
        caches.append(cache)

    WL = parameters['W' + str(L)]
    bL = parameters['b' + str(L)]
    ZL = WL @ A + bL
    AL = softmax(ZL)
    linear_cache = (A, WL, bL)
    caches.append((linear_cache, ZL))

    return AL, caches


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    dW = dZ @ A_prev.T
    db = np.reshape(np.squeeze(np.sum(dZ, axis=1, keepdims=True)), b.shape)
    dA_prev = W.T @ dZ

    return dA_prev, dW, db


def backpropagation(AL, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    A_prev, W, b = caches[-1][0]
    dAL, dWL, dbL = softmax_regression_grad(A_prev, W, b, Y)
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = dAL, dWL, dbL

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        A_prev, W, b = current_cache[0]
        Zl = current_cache[1]
        dAl = grads["dA" + str(l + 2)]
        grads["dA" + str(l + 1)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = linear_backward(
            (calc_tanh_grad(W @ A_prev + b)) * dAl, current_cache[0])

    return grads


def update_parameters(parameters, grads, lr):
    L = len(parameters) // 2  # number of layers

    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - lr * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - lr * grads["db" + str(l)]

    return parameters


def clasify(X, parameters):
    m = len(X[1])
    L = len(parameters) // 2
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


def SGD_nn(Xt, Yt, Xv, Yv, layers_dims, epochs, batch, lr):
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
        A_prev, W, b = caches[-1][0]
        costs.append(softmax_regression(A_prev, W, b, Yt))
        success_training.append(check_success(Xt, Yt, parameters))
        success_validation.append(check_success(Xv, Yv, parameters))

    return parameters, success_training, success_validation


def update_parameters_grad_test(parameters, parameters_D, eps):
    L = len(parameters) //2
    new_parameters = {}
    for l in range(1, L + 1):
        new_parameters["W" + str(l)] = parameters["W" + str(l)] + eps * parameters_D["W" + str(l)]
        new_parameters["b" + str(l)] = parameters["b" + str(l)] + eps * parameters_D["b" + str(l)]
    return new_parameters


def stack_w_b_grads(grads):
    stack = []
    l = len(grads) // 3
    for i in range(1, l + 1):
        stack.append(grads["dW" + str(i)].flatten())
        stack.append(grads["db" + str(i)].flatten())
    return np.concatenate(stack, axis=0)


def stack_parametersD(parameters_D):
    stack = []
    l = len(parameters_D) // 2
    for i in range(1, l + 1):
        stack.append(parameters_D["W" + str(i)].flatten())
        stack.append(parameters_D["b" + str(i)].flatten())
    return np.concatenate(stack, axis=0)


def jacobian_test_WB():
    X = np.random.randn(20, 1)
    W = np.random.randn(2, 20)
    b = np.random.randn(2, 1)

    v_u = np.random.randn(2, 1)
    u = v_u / np.linalg.norm(v_u)

    D_W = np.random.rand(2, 20)
    D_W = (1 / LA.norm(D_W)) * D_W

    D_b = np.random.rand(2, 1)
    D_b = (1 / LA.norm(D_b)) * D_b

    zero_loss = []
    one_loss = []
    epsilon = 1

    f0 = (np.tanh(W @ X + b).T @ u).flatten()
    dX, dW, db = linear_backward((calc_tanh_grad(W @ X + b)) * u, (X, W, b))

    for i in range(20):
        w_eps_k = W + epsilon * D_W
        b_eps_k = b + epsilon * D_b
        f1 = (np.tanh(w_eps_k @ X + b_eps_k).T @ u).flatten()
        zero_loss.append(abs(f1 - f0))
        one_loss.append(abs(
            f1 - f0 - (epsilon * (np.ndarray.flatten(D_W) @ np.ndarray.flatten(dW))) - (epsilon * (np.ndarray.flatten(D_b) @ np.ndarray.flatten(db)))))
        epsilon *= 0.5

    plot_grad_test(zero_loss, one_loss, 'Jacobian transpose test for the derivative of the layer by W and b')




def jacobian_test_X():
    X = np.random.randn(2, 1)
    W = np.random.rand(2, 2)
    b = np.random.rand(2, 1)

    v_u = np.random.randn(2, 1)
    u = v_u / np.linalg.norm(v_u)

    D_X = np.random.rand(2, 1)
    D_X = (1 / LA.norm(D_X)) * D_X

    zero_loss = []
    one_loss = []
    epsilon = 1
    f0 = (np.tanh(W @ X + b).T @ u).flatten()
    dX, dW, db = linear_backward((calc_tanh_grad(W @ X + b)) * u, (X, W, b))
    for i in range(20):
        fk = (np.tanh(W @ (X + epsilon * D_X) + b).T @ u).flatten()
        f1 = f0 + (epsilon * (np.ndarray.flatten(D_X) @ np.ndarray.flatten(dX)))
        zero_loss.append(abs(fk - f0))
        one_loss.append(abs(fk - f1))
        epsilon *= 0.5
    plot_grad_test(zero_loss, one_loss, 'Jacobian transpose test for the derivative of the layer by X')



def grad_Test_nn():
    X = Yt_Peaks
    Y = Ct_Peaks
    eps = 0.1
    linearly_grad_test = []
    quadratically_grad_test = []

    parameters = init_params([2, 3, 6, 5], D=0)
    parameters_D = init_params([2, 3, 6, 5], D=1)
    num_params = len(parameters) // 2

    AL, caches = forward_pass(X, parameters, np.tanh)
    grads = backpropagation(AL, Y, caches)

    f0 = softmax_regression(caches[-1][0][0], parameters["W" + str(num_params)], parameters["b" + str(num_params)],
                               Y)
    stack_grads = stack_w_b_grads(grads)
    stack_parameters_D = stack_parametersD(parameters_D)

    for i in range(20):
        new_parameters = update_parameters_grad_test(parameters, parameters_D, eps)
        AL, caches = forward_pass(X, new_parameters, np.tanh)
        fK = softmax_regression(caches[-1][0][0], new_parameters["W" + str(num_params)],
                                   new_parameters["b" + str(num_params)],
                                   Y)

        f1 = f0 + eps * stack_grads @ stack_parameters_D

        linearly_grad_test.append(abs(fK - f0))
        quadratically_grad_test.append(abs(fK - f1))

        eps *= 0.5

    plot_grad_test(linearly_grad_test, quadratically_grad_test, "Grad test for the whole Neural Network")



def test_data(Xt, Ct, Xv, Cv, hidden_layer, type, lr, batch):
    epochs = 100
    n = len(Xt)
    l = len(Ct)
    layer_dims = [n] + hidden_layer + [l]

    parameters, success_training, success_validation = SGD_nn(Xt, Ct, Xv, Cv, layer_dims, epochs, batch, lr)
    plot(success_training, success_validation, type, lr, batch, title="Neural Network using SGD")


def test_data_200(Xt, Ct, Xv, Cv, hidden_layer, type, lr, batch):
    epochs = 100
    n = len(Xt)
    l = len(Ct)
    layer_dims = [n] + hidden_layer + [l]
    X_200, C_200 = sample(Xt, Ct, 200)

    parameters, success_training, success_validation = SGD_nn(X_200, C_200, Xv, Cv, layer_dims, epochs, batch, lr)
    plot(success_training, success_validation, type, lr, batch, title="Neural Network using SGD - 200 training samples")





# test_data(Yt_SwissRoll, Ct_SwissRoll, Yv_SwissRoll, Cv_SwissRoll, [10, 10, 4], "Swiss Roll", lr=0.5, batch=100)
# test_data_200(Yt_SwissRoll, Ct_SwissRoll, Yv_SwissRoll, Cv_SwissRoll, [10, 10, 4], "Swiss Roll", lr=0.5, batch=100)
#
# test_data(Yt_Peaks, Ct_Peaks, Yv_Peaks, Cv_Peaks, [10, 10, 4], "Peaks", lr=0.1, batch=100)
# test_data_200(Yt_Peaks, Ct_Peaks, Yv_Peaks, Cv_Peaks, [10, 10, 4], "Peaks", lr=0.1, batch=100)
#
# test_data(Yt_GMM, Ct_GMM, Yv_GMM, Cv_GMM, [10, 10, 4], "GMM", lr=0.5, batch=100)
# test_data_200(Yt_GMM, Ct_GMM, Yv_GMM, Cv_GMM, [10, 10, 4], "GMM", lr=0.5, batch=100)

jacobian_test_WB()
jacobian_test_X()
grad_Test_nn()