"""
main.py — Pipeline completo: Proyecto C Audio NMF
Corre todo el proyecto de principio a fin con un solo comando
"""

import os # Para que las rutas de las carpetas funcionen igual en Windows, Mac o Linux
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg") # Para que guarde las gráficas directo como foto y no me abra ventanas
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import soundfile as sf

sys.path.insert(0, BCGD_DIR)
from algoritmoBCGDUnificado import bcgd_matrix_factorization

os.makedirs(DATA_PROC, exist_ok=True)

# Rutas
ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BCGD_DIR   = os.path.join(ROOT, "BCGD")
DATA_RAW   = os.path.join(ROOT, "data", "raw")
DATA_PROC  = os.path.join(ROOT, "data", "processed")
PLOTS_DIR  = os.path.join(ROOT, "data", "processed")




# BLOQUE 0 Sanity Check: verificación numérica del gradiente ---------------------------------------

def sanity_check():
    print("  BLOQUE 0 Sanity Check del Gradiente")

    m, n, k = 2, 2, 1
    X = np.array([[5., 3.], [2., 4.]])
    W = np.array([[1.], [2.]])
    H = np.array([[2., 1.]])

    grad_W_analytical = (W @ H - X) @ H.T

    epsilon = 1e-5
    grad_W_numerical = np.zeros_like(W)

    def loss_nmf(W_val, H_val):
        return 0.5 * np.sum((X - W_val @ H_val) ** 2)

    f0 = loss_nmf(W, H)
    for a in range(m):
        for b in range(k):
            E = np.zeros_like(W)
            E[a, b] = 1.0
            grad_W_numerical[a, b] = (loss_nmf(W + epsilon * E, H) - f0) / epsilon

    error_relativo = (np.linalg.norm(grad_W_analytical - grad_W_numerical) /
                      np.linalg.norm(grad_W_analytical))

    print(f"  Gradiente Analítico : {grad_W_analytical.ravel()}")
    print(f"  Gradiente Numérico  : {grad_W_numerical.ravel()}")
    print(f"  Error Relativo      : {error_relativo:.8e}")

    if error_relativo < 1e-4:
        print("  RESULTADO: ÉXITO — Error relativo < 1e-4")
    else:
        print("  RESULTADO: ERROR — Revisar la derivación")
        sys.exit(1)



# BLOQUE 1 Pipeline DSP: mezcla, STFT y split temporal ---------------------------------------------

def pipeline_dsp():
    print("  BLOQUE 1 Pipeline DSP")

    speech_path = os.path.join(DATA_RAW, "ejemplo_voz.wav")
    noise_path  = os.path.join(DATA_RAW, "ejemplo_ruido.wav")

    # Carga y mezcla
    speech, _ = librosa.load(speech_path, sr=16000)
    noise,  _ = librosa.load(noise_path,  sr=16000)
    min_len   = min(len(speech), len(noise))
    mix       = speech[:min_len] + noise[:min_len] * 0.4

    sf.write(os.path.join(DATA_PROC, "mix_output.wav"), mix, 16000)
    print(f"  Mezcla guardada ({min_len} muestras, {min_len/16000:.2f}s)")

    # STFT → magnitud + fase
    stft      = librosa.stft(mix, n_fft=2048, hop_length=512)
    X_mag     = np.abs(stft)
    X_phase   = np.exp(1j * np.angle(stft))
    print(f"  Espectrograma X: {X_mag.shape}  (F x T)")

    # Prueba de reconstrucción iSTFT
    audio_rec = librosa.istft(X_mag * X_phase, hop_length=512)
    sf.write(os.path.join(DATA_PROC, "prueba_reconstruccion.wav"), audio_rec, 16000)

    # Split temporal contíguo: 70 / 15 / 15
    T         = X_mag.shape[1]
    t_train   = int(T * 0.70)
    t_val     = t_train + int(T * 0.15)
    X_train   = X_mag[:, :t_train]
    X_val     = X_mag[:, t_train:t_val]
    X_test    = X_mag[:, t_val:]

    print(f"  Split: train={X_train.shape} | val={X_val.shape} | test={X_test.shape}")

    # Guardar matrices
    np.save(os.path.join(DATA_PROC, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_PROC, "X_val.npy"),   X_val)
    np.save(os.path.join(DATA_PROC, "X_test.npy"),  X_test)
    np.save(os.path.join(DATA_PROC, "X_phase.npy"), X_phase)
    print("  Matrices guardadas en data/processed/")

    return X_train, X_val, X_test, X_phase



# BLOQUE 2 Algorithm 2: evaluar H con W fijo (GD proyectado puro) ----------------------------------

def solve_H_eval(W_fixed, X_eval, k, steps=300, alphaH=1e-3):

    k_dim, T_eval = W_fixed.shape[1], X_eval.shape[1]
    np.random.seed(42)
    H_eval = np.random.uniform(0, 1 / np.sqrt(k), (k_dim, T_eval))
    loss_history = []
    for _ in range(steps):
        g      = W_fixed.T @ (W_fixed @ H_eval - X_eval)
        H_eval = np.maximum(H_eval - alphaH * g, 0)
        loss_history.append(0.5 * np.sum((W_fixed @ H_eval - X_eval) ** 2))
    return H_eval, loss_history


def compute_rmse(X, W, H):
    F, T = X.shape
    return np.linalg.norm(X - W @ H, "fro") / np.sqrt(F * T)


# BLOQUE 3 Entrenamiento con los 3 optimizadores (k=10) --------------------------------------------
def entrenar_y_evaluar(X_train, X_val, X_test):
    print("  BLOQUE 2 Entrenamiento y Evaluación (k=10)")

    k           = 10
    steps_train = 500
    steps_eval  = 300
    beta        = 0.9
    configs = {
        "gd":       {"alphaW": 1e-3, "alphaH": 1e-3},
        "momentum": {"alphaW": 1e-3, "alphaH": 1e-3},
        "nesterov": {"alphaW": 1e-5, "alphaH": 1e-5},
    }
    methods = ["gd", "momentum", "nesterov"]
    results = {}

    for method in methods:
        print(f"\n  [{method.upper()}]")

        W_learned, H_train, loss_hist = bcgd_matrix_factorization(
            X=X_train, k=k, steps=steps_train,
            alphaW=configs[method]["alphaW"],
            alphaH=configs[method]["alphaH"],
            method=method, beta=beta, lambd=0.0
        )

        rmse_train = compute_rmse(X_train, W_learned, H_train)

        H_val,  _ = solve_H_eval(W_learned, X_val,  k, steps_eval, alphaH=1e-3)
        H_test, _ = solve_H_eval(W_learned, X_test, k, steps_eval, alphaH=1e-3)

        rmse_val  = compute_rmse(X_val,  W_learned, H_val)
        rmse_test = compute_rmse(X_test, W_learned, H_test)

        print(f"    Loss final : {loss_hist[-1]:.4f}")
        print(f"    RMSE train : {rmse_train:.6f}")
        print(f"    RMSE val   : {rmse_val:.6f}")
        print(f"    RMSE test  : {rmse_test:.6f}")

        results[method] = {
            "W": W_learned, "H_train": H_train,
            "H_val": H_val, "H_test": H_test,
            "loss_history": loss_hist,
            "rmse_train": rmse_train,
            "rmse_val": rmse_val, "rmse_test": rmse_test,
        }

    # Resumen
    print(f"\n  {'Método':<12} {'RMSE Train':>12} {'RMSE Val':>10} {'RMSE Test':>10}")
    for m in methods:
        r = results[m]
        print(f"  {m:<12} {r['rmse_train']:>12.6f} {r['rmse_val']:>10.6f} {r['rmse_test']:>10.6f}")

    # Guardar matrices
    best = min(results, key=lambda m: results[m]["rmse_val"])
    print(f"\n  Mejor método por RMSE val: {best.upper()}")

    np.save(os.path.join(DATA_PROC, "W_best.npy"),       results[best]["W"])
    np.save(os.path.join(DATA_PROC, "H_train_best.npy"), results[best]["H_train"])
    np.save(os.path.join(DATA_PROC, "H_val_best.npy"),   results[best]["H_val"])
    np.save(os.path.join(DATA_PROC, "H_test_best.npy"),  results[best]["H_test"])
    for m in methods:
        np.save(os.path.join(DATA_PROC, f"W_{m}.npy"),         results[m]["W"])
        np.save(os.path.join(DATA_PROC, f"H_train_{m}.npy"),   results[m]["H_train"])
        np.save(os.path.join(DATA_PROC, f"loss_{m}.npy"),      np.array(results[m]["loss_history"]))

    return results


# BLOQUE 4 Experimentos: barrido de k y gráficas ---------------------------------------------------
def entrenar_con_controles(X, k, steps, alphaW, alphaH, method):
    
    W, H, loss_history = bcgd_matrix_factorization(
        X=X, k=k, steps=steps, alphaW=alphaW, alphaH=alphaH,
        method=method, lambd=0.0
    )
    loss_filtrada = []
    motivo = "Convergencia normal"
    for i, loss in enumerate(loss_history):
        if np.isnan(loss) or np.isinf(loss):
            motivo = "Divergencia (NaN)"
            break
        if i > 5:
            mejoras = [abs(loss_history[j] - loss_history[j-1]) for j in range(i-4, i+1)]
            if all(mj < 1e-4 for mj in mejoras):
                motivo = "Estancamiento"
                loss_filtrada.append(loss)
                break
        loss_filtrada.append(loss)
    return W, H, loss_filtrada, motivo


def experimentos_y_graficas(X_train, X_val, results_k10):
    print("  BLOQUE 3 Experimentos (k ∈ {5,10,15,20}) y Gráficas")

    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 11,
        "axes.titlesize": 14, "axes.labelsize": 12,
        "lines.linewidth": 2, "lines.markersize": 6,
        "figure.autolayout": True,
    })

    k_values   = [5, 10, 15, 20]
    metodos    = ["gd", "momentum", "nesterov"]
    iteraciones = 150
    resultados = {}

    print("\n  Barrido de hiperparámetro k...")
    for k in k_values:
        print(f"  k = {k}")
        resultados[k] = {}
        for metodo in metodos:
            alpha = 1e-5 if metodo == "nesterov" else 1e-3
            W, H, loss_hist, motivo = entrenar_con_controles(
                X_train, k, iteraciones, alpha, alpha, metodo
            )
            H_val, _ = solve_H_eval(W, X_val, k, steps=300, alphaH=1e-3)
            rmse_val  = compute_rmse(X_val, W, H_val)
            resultados[k][metodo] = {
                "W": W, "H": H,
                "loss_history": loss_hist,
                "rmse_val": rmse_val,
                "status": motivo,
            }
            print(f"    {metodo.upper():<10} [{motivo}]  RMSE val: {rmse_val:.4f}")

    # Gráfica 1: Pérdida vs iteración (k=10)
    k_viz = 10
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(resultados[k_viz]["gd"]["loss_history"],
            label="Vanilla GD", linestyle=":", color="#FF6B6B")
    ax.plot(resultados[k_viz]["momentum"]["loss_history"],
            label="Momentum",   linestyle="--", color="#4ECDC4")
    ax.plot(resultados[k_viz]["nesterov"]["loss_history"],
            label="Nesterov",   linewidth=2.5,  color="#292F36")
    ax.set_yscale("log")
    ax.set_title(f"Convergencia de la Función de Pérdida (k={k_viz})", fontweight="bold")
    ax.set_xlabel("Iteraciones")
    ax.set_ylabel("Pérdida (escala log)")
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    fig.savefig(os.path.join(PLOTS_DIR, "loss_vs_iter.png"), dpi=150)
    plt.close(fig)
    print("\n  ✓ Gráfica guardada: loss_vs_iter.png")

    # Gráfica 2-3: Heatmaps W y H (Nesterov, k=10)
    W_best = resultados[k_viz]["nesterov"]["W"]
    H_best = resultados[k_viz]["nesterov"]["H"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(W_best, ax=ax1, cmap="magma",   cbar=True)
    ax1.set_title(f"W: Bases Espectrales (k={k_viz})", fontweight="bold")
    ax1.set_xlabel("Componente (k)")
    ax1.set_ylabel("Bin de Frecuencia")
    sns.heatmap(H_best, ax=ax2, cmap="viridis", cbar=True)
    ax2.set_title(f"H: Activaciones Temporales (k={k_viz})", fontweight="bold")
    ax2.set_xlabel("Frame de Tiempo")
    ax2.set_ylabel("Componente (k)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "bases_activaciones.png"), dpi=150)
    plt.close(fig)
    print("  ✓ Gráfica guardada: bases_activaciones.png")

    # Gráfica 4: Val RMSE vs k
    k_list       = sorted(resultados.keys())
    rmse_val_gd  = [resultados[k]["gd"]["rmse_val"]       for k in k_list]
    rmse_val_mom = [resultados[k]["momentum"]["rmse_val"]  for k in k_list]
    rmse_val_nes = [resultados[k]["nesterov"]["rmse_val"]  for k in k_list]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_list, rmse_val_gd,  marker="o", label="GD",       color="#FF6B6B")
    ax.plot(k_list, rmse_val_mom, marker="s", label="Momentum", color="#4ECDC4")
    ax.plot(k_list, rmse_val_nes, marker="^", label="Nesterov", color="#292F36", markersize=8)

    for rmse_list, color in zip(
        [rmse_val_gd, rmse_val_mom, rmse_val_nes],
        ["#FF6B6B", "#4ECDC4", "#292F36"]
    ):
        k_opt    = k_list[np.argmin(rmse_list)]
        rmse_opt = min(rmse_list)
        ax.annotate(f"k={k_opt}", xy=(k_opt, rmse_opt),
                    xytext=(k_opt + 0.3, rmse_opt * 1.02),
                    fontsize=9, color=color)

    ax.set_title("Val RMSE vs. Rango $k$ por Optimizador", fontweight="bold")
    ax.set_xlabel("Número de Componentes ($k$)")
    ax.set_ylabel("RMSE de Validación")
    ax.set_xticks(k_list)
    ax.legend()
    ax.grid(axis="y", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "rmse_val_vs_k.png"), dpi=150)
    plt.close(fig)
    print("  ✓ Gráfica guardada: rmse_val_vs_k.png")

    # Tabla resumen
    print(f"\n  {'k':<6} {'RMSE GD':>12} {'RMSE Mom':>12} {'RMSE Nes':>12}")
    print("  " + "-"*44)
    for i, k in enumerate(k_list):
        print(f"  {k:<6} {rmse_val_gd[i]:>12.6f} {rmse_val_mom[i]:>12.6f} {rmse_val_nes[i]:>12.6f}")

    best_k = k_list[np.argmin(rmse_val_nes)]
    print(f"\n  k óptimo seleccionado (Nesterov): {best_k}")

    return resultados


# BLOQUE 5 Reconstrucción del audio separado con el mejor modelo -----------------------------------
def reconstruir_audio(results_k10, X_phase):
    print("  BLOQUE 4 Reconstrucción de Audio")

    best = min(results_k10, key=lambda m: results_k10[m]["rmse_val"])
    W    = results_k10[best]["W"]
    H    = results_k10[best]["H_train"]

    # Reconstrucción completa: X_hat = W @ H (magnitud separada)
    # Re-inyectamos la fase original de la mezcla
    X_phase_full = np.load(os.path.join(DATA_PROC, "X_phase.npy"))
    X_hat        = W @ H

    # Ajuste de dimensiones si el split acortó H
    T_phase = X_phase_full.shape[1]
    T_hat   = X_hat.shape[1]
    T_min   = min(T_phase, T_hat)

    complex_stft      = X_hat[:, :T_min] * X_phase_full[:, :T_min]
    audio_reconstruido = librosa.istft(complex_stft, hop_length=512)

    out_path = os.path.join(DATA_PROC, "audio_separado.wav")
    sf.write(out_path, audio_reconstruido, 16000)
    print(f"   Audio reconstruido guardado: audio_separado.wav")
    print(f"    Método usado: {best.upper()}")



# EJECUCIÓN DEL PIPELINE COMPLETO

print("  PROYECTO C — Audio NMF  |  Pipeline Completo")

# 0. Sanity check
sanity_check()

# 1. DSP: mezcla → STFT → split
X_train, X_val, X_test, X_phase = pipeline_dsp()

# 2. Entrenar 3 optimizadores con k=10 y evaluar
results_k10 = entrenar_y_evaluar(X_train, X_val, X_test)

# 3. Barrido de k y gráficas
experimentos_y_graficas(X_train, X_val, results_k10)

# 4. Reconstruir audio con el mejor modelo
reconstruir_audio(results_k10, X_phase)


print("  Pipeline completado. Resultados en data/processed/")
print("\n  Archivos generados:")
for f in sorted(os.listdir(DATA_PROC)):
    print(f"    {f}")