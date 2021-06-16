# -*- coding: utf8 -*-

import numpy as np

def viterbi(pi, a, b, obs):
    """Viterbi algorithm for solving the uncovering problem

    Notebook: C5/C5S3_Viterbi.ipynb

    Args:
        a (np.ndarray): State transition probability matrix of dimension I x I
        pi (np.ndarray): Initial state distribution  of dimension I
        b (np.ndarray): Output probability matrix of dimension I x K
        obs (np.ndarray): Observation sequence of length N

    Returns:
        path (np.ndarray): Optimal state sequence of length N
        delta (np.ndarray): Accumulated probability matrix
        E (np.ndarray): Backtracking matrix
    """
    n_states = a.shape[0]  # Number of states
    T = len(obs)  # Length of observation sequence

    # Initialize D and E matrices
    delta = np.zeros((n_states, T))
    E = np.zeros((n_states, T - 1)).astype(np.int32)
    delta[:, 0] = np.multiply(pi, b[:, obs[0]])

    # Compute D and E in a nested loop
    print('start walk forward'.center(30, '-'))
    for n in range(1, T):
        for i in range(n_states):
            temp_product = np.multiply(a[:, i], delta[:, n - 1])
            delta[i, n] = np.max(temp_product) * b[i, obs[n]]
            E[i, n - 1] = np.argmax(temp_product)
            print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=i, t=n, phi=E[i, n - 1]))

    # Backtracking
    print('start backtrace'.center(30, '-'))
    path = np.zeros(T).astype(np.int32)
    path[-1] = np.argmax(delta[:, -1])
    for n in range(T - 2, -1, -1):
        path[n] = E[int(path[n + 1]), n]
        print('path[{}] = {}'.format(n, path[n]))
    return path, delta, E