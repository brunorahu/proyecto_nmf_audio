"""
evaluacion_pruebas.py
---------------------
Módulo compartido con las funciones de evaluación del Proyecto C — Audio NMF.
Debe colocarse en la carpeta notebooks/ junto a los notebooks.

Exporta:
    solve_H_eval  — Algorithm 2 del enunciado (GD proyectado con W fijo)
    compute_rmse  — RMSE de reconstrucción espectral

Uso desde notebook 03:
    from evaluacion_pruebas import solve_H_eval, compute_rmse
"""

import numpy as np


def solve_H_eval(W_fixed, X_eval, k, steps=300, alphaH=1e-3):
    """
    Algoritmo 2 (enunciado): Resuelve para H_eval fijando el diccionario W aprendido.

    Implementa exactamente el pseudocódigo del Algorithm 2:
        g       <- W^T (W H_eval - X_eval)
        H_eval  <- max(H_eval - alphaH * g, 0)

    GD proyectado puro — sin momentum ni Nesterov. El subproblema en H
    con W fijo es convexo cuadrático, por lo que GD proyectado converge
    de forma monótona. Es el procedimiento de evaluación estándar del enunciado.

    Parámetros
    ----------
    W_fixed : ndarray (F, k)   Diccionario aprendido. Se mantiene fijo.
    X_eval  : ndarray (F, T)   Espectrograma de validación o prueba.
    k       : int              Número de componentes.
    steps   : int              Iteraciones de GD proyectado.
    alphaH  : float            Tasa de aprendizaje.

    Retorna
    -------
    H_eval       : ndarray (k, T)   Activaciones optimizadas.
    loss_history : list             Pérdida por iteración.
    """
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
    """
    RMSE de reconstrucción espectral: ||X - WH||_F / sqrt(F * T)

    Parámetros
    ----------
    X : ndarray (F, T)   Espectrograma original.
    W : ndarray (F, k)   Diccionario espectral.
    H : ndarray (k, T)   Activaciones temporales.

    Retorna
    -------
    rmse : float
    """
    F, T = X.shape
    return np.linalg.norm(X - W @ H, 'fro') / np.sqrt(F * T)
