# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np
from scipy.misc import logsumexp
import theano
import theano.tensor as T
import hypernet_trainer as ht
import ign.ign as ign
from ign.t_util import make_shared_dict, make_unshared_dict


def dm_example(N):
    x = 0.5 * np.random.rand(N, 1)

    std_dev = np.sqrt(0.02)
    noise = std_dev * np.random.randn(N, 1)
    x_n = x + noise
    y = x + 0.3 * np.sin(2.0 * np.pi * x_n) + 0.3 * np.sin(4.0 * np.pi * x_n) \
        + noise
    return x, y


def unpack(v, weight_shapes):
    L = []
    tt = 0
    for ws in weight_shapes:
        num_param = np.prod(ws)
        L.append(v[tt:tt + num_param].reshape(ws))
        tt += num_param
    return L


def loglik_primary_f(X, y, theta, weight_shapes):
    W0, b0, W1, b1 = unpack(theta, weight_shapes)

    a1 = T.maximum(0.0, T.dot(X, W0) + b0[None, :])
    yp = T.dot(a1, W1) + b1[None, :]
    err = yp - y

    # Assuming std=1 here
    loglik = -0.5 * T.sum(err ** 2, axis=1)  # Ignoring normalizing constant
    return loglik


def logprior_f(theta):
    # Standard Gauss
    logprior = -0.5 * T.sum(theta ** 2)  # Ignoring normalizing constant
    return logprior


def simple_test(X, y, X_valid, y_valid,
                n_epochs, n_batch, init_lr, weight_shapes,
                n_layers=5, vis_freq=100, n_samples=100):
    N, D = X.shape
    N_valid = X_valid.shape[0]
    assert(y.shape == (N, 1))  # Univariate for now
    assert(X_valid.shape == (N_valid, D) and y_valid.shape == (N_valid, 1))

    num_params = sum(np.prod(ws) for ws in weight_shapes)

    layers = ign.init_ign(n_layers, num_params)
    phi_shared = make_shared_dict(layers, '%d%s')

    ll_primary_f = lambda X, y, w: loglik_primary_f(X, y, w, weight_shapes)
    hypernet_f = lambda z: ign.network_T_and_J(z[None, :], phi_shared)[0][0, :]
    # TODO verify this length of size 1
    log_det_dtheta_dz_f = lambda z: T.sum(ign.network_T_and_J(z[None, :], phi_shared)[1])

    R = ht.build_trainer(phi_shared.values(), N, ll_primary_f, logprior_f, hypernet_f,
                         log_det_dtheta_dz_f=log_det_dtheta_dz_f)
    trainer, get_err, test_loglik = R

    batch_order = np.arange(int(N / n_batch))

    cost_hist = np.zeros(n_epochs)
    loglik_valid = np.zeros(n_epochs)
    for epoch in xrange(n_epochs):
        np.random.shuffle(batch_order)

        cost = 0.0
        current_lr = init_lr
        for ii in batch_order:
            x_batch = X[ii * n_batch:(ii + 1) * n_batch]
            y_batch = y[ii * n_batch:(ii + 1) * n_batch]
            z_noise = np.random.randn(num_params)
            batch_cost = trainer(x_batch, y_batch, z_noise, current_lr)
            cost += batch_cost
        cost /= len(batch_order)
        print cost
        cost_hist[epoch] = cost

        loglik_valid_s = np.zeros((N_valid, n_samples))
        for ss in xrange(n_samples):
            z_noise = np.random.randn(num_params)
            loglik_valid_s[:, ss] = test_loglik(X_valid, y_valid, z_noise)
        loglik_valid[epoch] = np.mean(logsumexp(loglik_valid_s - np.log(n_samples), axis=1))
        print 'valid %f' % loglik_valid[epoch]

    phi = make_unshared_dict(phi_shared)
    return phi, cost_hist, loglik_valid


if __name__ == '__main__':
    np.random.seed(5645)

    init_lr = 0.001
    n_epochs = 500
    n_batch = 32
    N = 1000

    primary_layers = 1
    input_dim = 1
    hidden_dim = 10
    output_dim = 1

    weight_shapes = [(input_dim, hidden_dim), (hidden_dim,),
                     (hidden_dim, output_dim), (output_dim,)]

    X, y = dm_example(N)
    X_valid, y_valid = dm_example(N)
    phi, cost_hist, loglik_valid = \
        simple_test(X, y, X_valid, y_valid,
                    n_epochs, n_batch, init_lr, weight_shapes,
                    n_layers=2)
