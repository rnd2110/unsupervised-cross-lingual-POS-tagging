### The script is a based on https://github.com/afshinrahimi/mmner/blob/master/bea_code/run_bea_conll.ipynb ###

import numpy as np
from scipy.special import digamma

def format_e(n):
    return "{:.2f}".format(n)

# Weighted maximum voting -- Can you be used separately or to initialize the Bayesian inference.
def mv_infer(values, weights, num_classes):
    num_items, num_workers = values.shape
    all_items = np.arange(num_items)
    z_ik = np.zeros((num_items, num_classes))

    if len(weights) == 0:
        for j in range(num_workers):
            z_ik[all_items, values[:, j]] += 1
    else:
        for i in range(num_items):
            for j in range(num_workers):
                z_ik[i, values[i, j]] += (1 * weights[i, j])

    return z_ik, 1

# Bayesian inference
def bea_infer(values, weights, num_classes, alpha=1, beta_kl=1, prior=True):

    num_items, num_workers = values.shape

    beta_kl = beta_kl * np.ones((num_classes, num_classes))

    z_ik, it = mv_infer(values, weights, num_classes)
    n_jkl = np.empty((num_workers, num_classes, num_classes))

    last_z_ik = z_ik.copy()

    # Optimization iterations (up to 500 in the case of not converging)
    for iteration in range(500):
        Eq_log_pi_k = digamma(z_ik.sum(axis=0) + alpha) - digamma(num_items + num_classes * alpha)

        n_jkl[:] = beta_kl
        for j in range(num_workers):
            for k in range(num_classes):
                n_jkl[j, k, :] += np.bincount(values[:, j], z_ik[:, k], minlength=num_classes)
        Eq_log_v_jkl = digamma(n_jkl) - digamma(n_jkl.sum(axis=-1, keepdims=True))

        if prior:
            z_ik[:] = Eq_log_pi_k
        else:
            z_ik.fill(0)

        for j in range(num_workers):
            z_ik += Eq_log_v_jkl[j, :, values[:, j]]

        z_ik -= z_ik.max(axis=-1, keepdims=True)
        z_ik = np.exp(z_ik)
        z_ik /= z_ik.sum(axis=-1, keepdims=True)

        if np.allclose(z_ik, last_z_ik, atol=1e-3):
            break

        last_z_ik[:] = z_ik

    return z_ik, iteration