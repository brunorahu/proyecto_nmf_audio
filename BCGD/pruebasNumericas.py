import numpy as np

#Configuración del ejemplo pequeño (m=2, n=2, k=1) con valores enteros.
m, n, k = 2, 2, 1
#Matrices inicializadas con valores pequeños arbitrarios.
X = np.array([[5., 3.], 
              [2., 4.]])
W = np.array([[1.], 
              [2.]])
H = np.array([[2., 1.]])

#1. Cálculo del gradiente analítico utilizando la fórmula matricial derivada: (WH - X)H^T.
grad_W_analytical = (W @ H - X) @ H.T

#2. Verificación numérica utilizando la aproximación de diferencias finitas.
epsilon = 1e-5
grad_W_num = np.zeros_like(W)
#Definición de la pérdida base sin enmascarar.
def f_base(W_val, H_val):
    return 0.5 * np.sum((X - W_val @ H_val)**2)
#Llenado entrada por entrada del gradiente aproximado
for a in range(m):
    for b in range(k):
        #Matriz E_ab: tiene un 1 en la posición (a,b) y ceros en el resto.
        E = np.zeros_like(W)
        E[a, b] = 1
        #Fórmula de diferencias finitas hacia adelante.
        grad_W_num[a, b] = (f_base(W + epsilon * E, H) - f_base(W, H)) / epsilon
        
#Cálculo del error relativo entre ambos gradientes.
error_relativo = np.linalg.norm(grad_W_analytical - grad_W_num) / np.linalg.norm(grad_W_analytical)
print("=== Verificación de Gradientes (Sanity Check) ===")
print(f"Gradiente Analítico:\n{grad_W_analytical}")
print(f"Gradiente Numérico:\n{grad_W_num}")
print(f"Error Relativo: {error_relativo:.8e}")

#Comprobación de éxito requerida en el reporte.
if error_relativo < 1e-4:
    print("¡Éxito! El error relativo está por debajo de 1e-4, la derivación teórica es correcta.")
else:
    print("Error: Revisa la derivación o la implementación, el error es demasiado alto.")