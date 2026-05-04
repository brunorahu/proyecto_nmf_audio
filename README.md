Proyecto C: Separacion de Fuentes de Audio mediante NMF y BCGD
==============================================================

Optimizacion Matemática  - Cuarto Semestre - 04 de Mayo 2026

Autores
-------
Diego de Jesus Munoz Gonzalez
Alexander Gongora Venegas
Bruno Gael Ramos Huerta
Julio Alfonso Rangel Ortiz


Descripcion General
-------------------
Este proyecto implementa separacion de fuentes de audio usando Factorizacion de
Matrices No Negativas (NMF) optimizada mediante Descenso de Gradiente por
Coordenadas en Bloques (BCGD). Dado el espectrograma de magnitud de una mezcla
de voz y ruido, el modelo aprende bases espectrales W y activaciones temporales H
tales que X aprox W*H, permitiendo reconstruir las fuentes por separado via iSTFT.

Dataset: MUSAN (Music, Speech, and Noise Corpus)
Referencia: Lee & Seung (1999). Nature, 401(6755), 788-791.


Estructura del Proyecto
-----------------------
```
PROYECTO_NMF_AUDIO/
    BCGD/
        algoritmoBCGDUnificado.py   Algoritmo 1: BCGD unificado (GD, Momentum, Nesterov)
        pruebasNumericas.py         Sanity check numerico del gradiente
        README.txt                  Este archivo
    data/
        raw/
            ejemplo_voz.wav         Audio de voz (MUSAN)
            ejemplo_ruido.wav       Audio de ruido (MUSAN)
        processed/                  Generado automaticamente al correr main.py
    notebooks/
        main.py                     Pipeline completo (punto de entrada unico)
        evaluacion_pruebas.py       Modulo compartido: solve_H_eval y compute_rmse
        01_pipeline_dsp.ipynb       Pipeline DSP paso a paso
        02_evaluacion_pruebasV1.ipynb   Evaluacion y comparacion de optimizadores
        03_experimentos_visualizacionV1.ipynb   Experimentos por k y graficas
        EDA.ipynb                   Analisis exploratorio de datos

```
Requisitos
----------

Instalar dependencias:
```
    pip install numpy librosa soundfile matplotlib seaborn
```


Se recomienda usar un entorno virtual:
```
    python -m venv .venv
    source .venv/bin/activate        (Linux / Mac)
    .venv\Scripts\activate           (Windows)
    pip install numpy librosa soundfile matplotlib seaborn
```

Como Correr el Proyecto (pipeline completo)
-------------------------------------------
Desde la carpeta notebooks/, ejecutar:

    python main.py

El script corre todo el pipeline de principio a fin sin intervencion manual:

    Bloque 0 - Sanity check del gradiente
        Verifica numericamente que las derivadas analiticas son correctas.
        Error relativo obtenido: 3.33e-06 (umbral requerido: 1e-04).
        Si el error supera el umbral el script termina con un mensaje de error.

    Bloque 1 - Pipeline DSP
        Carga los archivos de voz y ruido desde data/raw/.
        Mezcla digitalmente las senales (ruido atenuado al 40%).
        Calcula la STFT (n_fft=2048, hop_length=512) y extrae la magnitud.
        Aplica el split temporal contiguo 70% train / 15% val / 15% test.
        Guarda X_train.npy, X_val.npy, X_test.npy y X_phase.npy.

    Bloque 2 - Entrenamiento y evaluacion (k=10)
        Entrena el modelo con los tres optimizadores (GD, Momentum, Nesterov).
        Evalua en validacion y prueba usando el Algoritmo 2 (GD proyectado puro).
        Imprime tabla comparativa de RMSE train / val / test.
        Guarda W y H de cada metodo, y determina el mejor por RMSE val.

    Bloque 3 - Experimentos y graficas
        Barre k en {5, 10, 15, 20} con los tres optimizadores.
        Genera tres graficas en data/processed/:
            loss_vs_iter.png        Convergencia de la perdida (escala log)
            bases_activaciones.png  Heatmaps de W y H
            rmse_val_vs_k.png       Val RMSE vs rango k (seleccion de hiperparametro)

    Bloque 4 - Reconstruccion de audio
        Toma el mejor modelo (por RMSE val) y reconstruye el audio via iSTFT.
        Guarda audio_separado.wav en data/processed/.

Archivos generados en data/processed/ al terminar:
    X_train.npy, X_val.npy, X_test.npy, X_phase.npy
    W_best.npy, H_train_best.npy, H_val_best.npy, H_test_best.npy
    W_gd.npy, W_momentum.npy, W_nesterov.npy
    H_train_gd.npy, H_train_momentum.npy, H_train_nesterov.npy
    loss_gd.npy, loss_momentum.npy, loss_nesterov.npy
    mix_output.wav, prueba_reconstruccion.wav, audio_separado.wav
    loss_vs_iter.png, bases_activaciones.png, rmse_val_vs_k.png


Descripcion de Archivos
-----------------------
algoritmoBCGDUnificado.py
    Contiene bcgd_matrix_factorization(), que implementa el Algoritmo 1 completo.
    Soporta los tres metodos de optimizacion: gd, momentum y nesterov.
    Parametros principales:
        X           Matriz de datos (espectrograma de magnitud)
        k           Numero de componentes latentes
        steps       Iteraciones del bucle externo
        innerW      Pasos internos para el bloque W (default: 1)
        innerH      Pasos internos para el bloque H (default: 1)
        alphaW      Tasa de aprendizaje para W
        alphaH      Tasa de aprendizaje para H
        method      'gd', 'momentum' o 'nesterov'
        beta        Coeficiente de momentum (default: 0.9)
        proj        Funcion de proyeccion (default: max(x, 0) para NMF)
        M           Mascara binaria opcional (default: matriz de unos)
        lambd       Regularizacion L2 (default: 0.0 para Audio NMF)
    Retorna: W, H, loss_history

pruebasNumericas.py
    Script de auditoria que compara el gradiente analitico derivado a mano
    contra una aproximacion de diferencias finitas (epsilon = 1e-5) en un
    ejemplo de dimensiones m=2, n=2, k=1. El error relativo resultante es
    3.33e-06, confirmando la correctitud de la derivacion.

evaluacion_pruebas.py
    Modulo compartido que exporta:
        solve_H_eval(W_fixed, X_eval, k, steps, alphaH)
            Implementa el Algoritmo 2: dado W fijo, resuelve para H_eval
            mediante GD proyectado puro (sin momentum ni Nesterov).
            El subproblema en H con W fijo es cuadratico convexo, por lo
            que GD proyectado converge de forma monotona.
        compute_rmse(X, W, H)
            Calcula ||X - WH||_F / sqrt(F * T).

main.py
    Punto de entrada unico. Ejecuta los cuatro bloques en orden e imprime
    el progreso en consola. No requiere argumentos.


Parametros del Modelo
---------------------
Parametro           Valor usado     Rango recomendado (enunciado)
k (barrido)         5, 10, 15, 20   5 a 20
steps (train)       500             200 a 500
steps (eval)        300             -
innerW / innerH     1 / 1           1
alpha (GD/Mom)      1e-3            1e-3
alpha (Nesterov)    1e-5            reducir si diverge
beta                0.9             0.9
lambda              0.0             0 para Audio NMF
proj                max(x, 0)       max(x, 0) para NMF


Resultados Principales
----------------------
Comparacion de optimizadores con k=10, 150 iteraciones:

    Metodo        RMSE Train    RMSE Val    RMSE Test    Convergencia
    Vanilla GD    6.484         75.491      84.531       Diverge
    Momentum      1.754         1.410       1.613        Oscila
    Nesterov      1.341         1.261       1.517        Monotona (mejor)

Seleccion de hiperparametro k por RMSE val (Nesterov):

    k=5   RMSE val: 1.50
    k=10  RMSE val: 1.38
    k=15  RMSE val: 1.30
    k=20  RMSE val: 1.13  <- optimo seleccionado

El modelo final usa Nesterov con k=20.


Notas Tecnicas
--------------
- El split temporal es contiguo (no aleatorio) para evitar fuga de datos
  entre frames adyacentes del espectrograma, que estan altamente correlacionados.

- GD vanilla diverge con alpha=1e-3 porque ese paso supera la constante de
  Lipschitz del gradiente en dimension F=1025. La politica de correccion es
  reducir alpha al menos dos ordenes de magnitud o usar Nesterov.

- La reconstruccion de audio via iSTFT usa la fase de la mezcla original.
  Esto introduce artefactos menores pero es el metodo estandar sin necesidad
  de estimacion iterativa de fase.

- La semilla aleatoria np.random.seed(42) garantiza reproducibilidad en todas
  las inicializaciones de W y H.


Referencias
-----------
[1] Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by
    non-negative matrix factorization. Nature, 401(6755), 788-791.
    https://doi.org/10.1038/44565

[2] Nesterov, Y. (1983). A method for solving a convex programming problem
    with convergence rate O(1/k^2). Soviet Mathematics Doklady, 27, 372-376.

[3] Snyder, D., Chen, G., & Povey, D. MUSAN: A Music, Speech, and Noise
    Corpus. arXiv:1510.08484.

[4] Virtanen, T. (2007). Monaural sound source separation by nonnegative
    matrix factorization. IEEE Trans. Audio, Speech, Language Process.,
    15(3), 1066-1074.
