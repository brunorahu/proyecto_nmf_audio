
# Funciones auxiliares para el Algorithm 2 y cálculo de RMSE

import numpy as np


def solve_H_eval(W_fixed, X_eval, k, steps=300, alphaH=1e-3):
    # Implementa el Algoritmo 2 (GD proyectado con W fijo) para obtener las activaciones H_eval.

    k_dim = W_fixed.shape[1]
    T_eval = X_eval.shape[1]

    np.random.seed(42)
    H_eval = np.random.uniform(0, 1 / np.sqrt(k), (k_dim, T_eval))

    loss_history = []

    for _ in range(steps):
        g = W_fixed.T @ (W_fixed @ H_eval - X_eval)
        H_eval = np.maximum(H_eval - alphaH * g, 0)
        loss_history.append(0.5 * np.sum((W_fixed @ H_eval - X_eval) ** 2))

    return H_eval, loss_history


def compute_rmse(X, W, H):
    # Calcula el error (RMSE) entre el espectrograma original y la reconstrucción W@H.
    F, T = X.shape
    return np.linalg.norm(X - W @ H, 'fro') / np.sqrt(F * T)
