1. algoritmoBCGDUnificado.py (El Motor Principal) 
¿Qué hace?: Este script contiene la función bcgd_matrix_factorization, la cual 
implementa el Algoritmo 1 completo. Su objetivo es tomar la matriz de datos original 
(el espectrograma X) y factorizarla de manera iterativa en las matrices W y H utilizando 
el Descenso de Gradiente por Coordenadas en Bloques (BCGD). Está diseñado para 
soportar las tres variantes de optimización requeridas en el proyecto: Vanilla GD, 
Momentum y Nesterov. 

2. pruebasNuméricas.py (Verificación Numérica) 
¿Qué hace? Es un script de auditoría o sanity check. El proyecto exige comprobar 
numéricamente que las derivadas analíticas calculadas a mano no tienen errores 
antes de implementarlas en el modelo principal. Este código toma un ejemplo con 
matrices de dimensiones pequeñas (m=2, n=2, k=1) y compara la fórmula del gradiente 
contra una aproximación de diferencias finitas . 