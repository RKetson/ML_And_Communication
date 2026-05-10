import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


from libs.val_model import train, recover_weights, aval_model, recover_points_model
from libs.topology import Net_Coded
from libs.model_E2E import End2EndSystem
from libs.AFF3CT_to_points import txt_to_dict

"""
    Experimento: Autoencoder com Maximização da Informação Mútua de Bit (BMI)

    Arquitetura (por símbolo de m=4 bits):
      Transmissor: one-hot(M=16) → Dense(n, linear, sem bias) → EnergyNormalization
      Canal:       AWGN
      Receptor:    Dense(2^(m+a), relu) → Dense(m, linear) — saída em logits (LLRs)

    O parâmetro `a` controla a capacidade do receptor:
        a=0 → 2^4 = 16 neurônios  (receptor mínimo)
        a=1 → 2^5 = 32 neurônios
        a=2 → 2^6 = 64 neurônios
        a=3 → 2^7 = 128 neurônios

    Uso:
        python Compare_BMI_FL_coded.py           # Carrega resultados salvos
        python Compare_BMI_FL_coded.py --retrain # Força novo treinamento
"""

# ============================================================================================ #
# Argumentos de linha de comando
# ============================================================================================ #
parser = argparse.ArgumentParser(description="BMI Autoencoder — Estudo do impacto da capacidade do receptor")
parser.add_argument("--retrain", action="store_true",
                    help="Força novo treinamento, sobrescrevendo resultados salvos.")
args = parser.parse_args()

FORCE_RETRAIN = args.retrain

# ============================================================================================ #
# Configuração de GPUs
# ============================================================================================ #
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    strategy = tf.distribute.get_strategy()
    print(f"Treinamento distribuído em {strategy.num_replicas_in_sync} GPU(s).")
else:
    strategy = tf.distribute.get_strategy()
    print("Nenhuma GPU encontrada. Usando CPU.")

# ============================================================================================ #
# Parâmetros do sistema
# ============================================================================================ #
BATCH_SIZE           = 2500
NUM_TRAINING_ITER    = 8000

k           = 4          # Bits de informação por símbolo
n           = 7          # Dimensões reais do símbolo transmitido (I e Q)
SNRdb_train = 5.0        # SNR de treinamento (dB)
ebno_dbs    = np.arange(-4, 8, 1)

# Valores de `a` a comparar
A_VALUES = [0, 1, 2, 3, 4, 5]

# ============================================================================================ #
# Diretórios de saída
# ============================================================================================ #
BUFFER_DIR = "./Buffer/Fully Connected/BMI"
PONTOS_DIR = "./Pontos/Autoencoder/Fully Connected/BMI"
FIG_DIR    = "./Figures/Fully Connected/Autoencoder-BMI"

os.makedirs(BUFFER_DIR, exist_ok=True)
os.makedirs(PONTOS_DIR, exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)

# ============================================================================================ #
# Treinamento e avaliação para cada valor de `a`
# ============================================================================================ #
results = {}  # {a: (ber_list, ser_list)}

for a in A_VALUES:
    print(f"\n{'='*60}")
    print(f" BMI Autoencoder — a={a}  (receptor: {2**(k+a)} neurônios)")
    print(f"{'='*60}")

    local_weights = f"{BUFFER_DIR}/weights-bmi-k{k}-n{n}-a{a}"
    local_aval    = f"{BUFFER_DIR}/constellation-bmi-k{k}-n{n}-a{a}"
    local_ber_ser = f"{PONTOS_DIR}/BER_SER-bmi-k{k}-n{n}-a{a}"

    with strategy.scope():
        bmi_tx = Net_Coded.encoder(k, n)
        bmi_rx = Net_Coded.decoder(k, n, a=a, bmi=True)

        bmi_model_train = End2EndSystem(k, n, bmi_tx, bmi_rx, training=True,  bit_wise=True)
        bmi_model       = End2EndSystem(k, n, bmi_tx, bmi_rx, training=False, bit_wise=True)

        # Taxa de aprendizado decrescente (Cosine Decay)
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-3,
            decay_steps=NUM_TRAINING_ITER,
            alpha=1e-5   # LR mínima ao final do treino
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # Treinamento
        if FORCE_RETRAIN or not os.path.exists(local_weights):
            train(bmi_model_train, SNRdb_train, optimizer, NUM_TRAINING_ITER, BATCH_SIZE,
                local_weights, aval_training=True, steps_for_aval=2500, local_aval=local_aval)

        # Recupera pesos
        bmi_model = recover_weights(bmi_model, local_weights)

    # Avaliação Monte Carlo
    if FORCE_RETRAIN or not os.path.exists(local_ber_ser):
        aval_model(bmi_model, ebno_dbs, max_iter=75000, block_errors=500,
                   graph_mode="xla", local=local_ber_ser)

    ber_dict, ser_dict = recover_points_model(local_ber_ser)
    ber = [ber_dict[ebno] for ebno in ebno_dbs]
    ser = [ser_dict[ebno] for ebno in ebno_dbs]
    results[a] = (ber, ser)

# ============================================================================================ #
# Referência: Hamming(7,4)
# ============================================================================================ #
ber_ref, ser_ref = txt_to_dict("./Pontos/AFF3CT/Hamming-7-4.txt")
ber_ref = [ber_ref.get(ebno, float('nan')) for ebno in ebno_dbs]
ser_ref = [ser_ref.get(ebno, float('nan')) for ebno in ebno_dbs]

ber_ref_MLD, ser_ref_MLD = txt_to_dict("./Pontos/AFF3CT/Hamming-7-4-MLD.txt")
ber_ref_MLD = [ber_ref_MLD.get(ebno, float('nan')) for ebno in ebno_dbs]
ser_ref_MLD = [ser_ref_MLD.get(ebno, float('nan')) for ebno in ebno_dbs]

# ============================================================================================ #
# Plot: BER e SER
# ============================================================================================ #
markers = ['o', 's', '^', 'D', 'x', '*']
colors  = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#3d405b', '#6a4c93']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'Autoencoder BMI (Bit-wise) vs Hamming ({n},{k})', fontsize=16, fontweight='bold')

for ax, metric_idx, ylabel, title in [
    (axes[0], 0, 'BER', 'Bit Error Rate'),
    (axes[1], 1, 'SER', 'Symbol Error Rate'),
]:
    # Curvas do autoencoder BMI para diferentes capacidades do receptor
    for i, a in enumerate(A_VALUES):
        ber, ser = results[a]
        values = ber if metric_idx == 0 else ser
        ax.semilogy(ebno_dbs, values,
                    marker=markers[i], color=colors[i], linewidth=1.8, markersize=6,
                    label=f'BMI a={a}  ({2**(k+a)} neurônios)')

    # Referência: Hamming(7,4) com Soft Decision
    ref = ber_ref_MLD if metric_idx == 0 else ser_ref_MLD
    ax.semilogy(ebno_dbs, ref, 'k--', linewidth=2.0, marker='x', markersize=7,
                label='Hamming(7,4) com Soft Decision')

    # Referência: Hamming(7,4) com Hard Decision
    ref = ber_ref if metric_idx == 0 else ser_ref
    ax.semilogy(ebno_dbs, ref, 'k-', linewidth=2.0, marker='x', markersize=7,
                label='Hamming(7,4) com Hard Decision')

    ax.set_xlabel('Eb/N0 (dB)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f"BMI_k{k}_n{n}_capacity_comparison.png"), dpi=150)
plt.show()

print(f"\nFigura salva em: {FIG_DIR}/BMI_k{k}_n{n}_capacity_comparison.png")
