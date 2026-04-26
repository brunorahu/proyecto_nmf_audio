import numpy as np
def bcgd_matrix_factorization(X, k, steps, innerW=1, innerH=1, 
                              alphaW=1e-3, alphaH=1e-3, method='gd', 
                              beta=0.9, proj=lambda x: np.maximum(x, 0), 
                              M=None, lambd=0.0):
    """
    Descenso de Gradiente por Coordenadas en Bloques (BCGD) Unificado para Factorización de Matrices.
    Implementa el algoritmo central del proyecto, diseñado para funcionar con diferentes dominios
    dependiendo de los hiperparámetros y la función de proyección.
    """
    m, n = X.shape
    #Máscara opcional. Si no se provee, asumimos una matriz de unos (sin máscara).
    if M is None:
        M = np.ones((m, n))

    #Paso 1: Inicialización aleatoria pequeña y positiva.
    #Se utiliza una semilla para asegurar que los experimentos sean reproducibles.
    #Se utiliza un escalamiento de 1/sqrt(k) para controlar la magnitud.
    np.random.seed(42) 
    W = np.random.uniform(0, 1/np.sqrt(k), (m, k))
    H = np.random.uniform(0, 1/np.sqrt(k), (k, n))
    
    #Paso 2: Inicialización de las matrices de velocidad en ceros.
    vW = np.zeros((m, k))
    vH = np.zeros((k, n))
    #Lista para almacenar la historia de la función de pérdida.
    loss_history = []
    
    #Paso 3: Bucle principal de optimización sobre todos los pasos.
    for s in range(steps):
        #Actualización del bloque W (manteniendo H fijo)
        for t in range(innerW):
            if method == 'nesterov':
                #Regla de Aceleración de Nesterov (Look-ahead) 
                W_look = W - alphaW * beta * vW
                R_look = M * (W_look @ H - X)
                gW_look = R_look @ H.T + lambd * W_look
                vW = beta * vW + gW_look # [cite: 156]
                W = W - alphaW * vW # [cite: 161]
            else:
                #Cálculo estándar del residual y gradiente base 
                R = M * (W @ H - X)
                gW = R @ H.T + lambd * W
                if method == 'gd':
                    #Vanilla Gradient Descent 
                    W = W - alphaW * gW
                elif method == 'momentum':
                    #Descenso con Momentum 
                    vW = beta * vW + gW
                    W = W - alphaW * vW
            #Proyección del bloque W. Para Audio NMF, esto fuerza la no negatividad
            W = proj(W)
        #Actualización del bloque H (manteniendo W fijo)
        for t in range(innerH):
            if method == 'nesterov':
                #Regla de Nesterov aplicada al bloque H, calculando sobre una posición adelantada [cite: 173]
                H_look = H - alphaH * beta * vH
                R_look = M * (W @ H_look - X)
                gH_look = W.T @ R_look + lambd * H_look
                vH = beta * vH + gH_look
                H = H - alphaH * vH
            else:
                #Cálculo estándar del residual y gradiente [cite: 169-171]
                R = M * (W @ H - X)
                gH = W.T @ R + lambd * H
                if method == 'gd':
                    H = H - alphaH * gH # [cite: 173]
                elif method == 'momentum':
                    vH = beta * vH + gH # [cite: 173]
                    H = H - alphaH * vH # [cite: 173]
            #Proyección del bloque H para asegurar las restricciones del proyecto[cite: 175].
            H = proj(H)
            
        #Paso 4: Cálculo de la pérdida (Loss) en la iteración actual.
        #La suma de los cuadrados se calcula de manera eficiente en numpy.
        residual = M * (W @ H - X)
        loss = 0.5 * np.sum(residual**2) + (lambd / 2) * (np.sum(W**2) + np.sum(H**2))
        loss_history.append(loss)
        
    return W, H, loss_history