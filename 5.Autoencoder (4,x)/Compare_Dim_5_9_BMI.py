import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from libs.val_model import recover_weights, aval_model, recover_points_model, train_curriculum
from libs.topology import Net_Coded
from libs.model_E2E import End2EndSystem
from libs.AFF3CT_to_points import txt_to_dict

"""
    Experimento: 5.Autoencoder (4,x)
    Objetivo: Analisar o desempenho de uma rede BMI (a=4) para dimensionalidade
    do transmissor (n) variando entre 5 e 9, utilizando métricas de BER e SER.

    Uso:
        python Compare_Dim_5_9_BMI.py           # Carrega resultados salvos
        python Compare_Dim_5_9_BMI.py --retrain # Força novo treinamento
"""

# ============================================================================================ #
# Argumentos de linha de comando
# ============================================================================================ #
parser = argparse.ArgumentParser(description="Desempenho BMI variando dimensão n")
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
BATCH_SIZE           = 5000
NUM_TRAINING_ITER    = 150000

k           = 4          # Bits de informação por símbolo
a           = 4          # Capacidade Fixa para BMI
ebno_dbs    = np.arange(-4, 8, 1)

n_values = [5, 6, 7, 8, 9]

# ============================================================================================ #
# Diretórios de saída
# ============================================================================================ #
MODELS_DIR        = "./Modelos/Pesos/FullyConnected/5.Autoencoder(4,x)"
CONSTELLATION_DIR = "./Modelos/Constelações/FullyConnected/5.Autoencoder(4,x)"
PONTOS_DIR        = "./Pontos/Autoencoder/Fully Connected/5.Autoencoder(4,x)"
FIG_DIR           = "./Figures/Fully Connected/5.Autoencoder(4,x)"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CONSTELLATION_DIR, exist_ok=True)
os.makedirs(PONTOS_DIR, exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)

results = {}  # {n: (ber_list, ser_list)}

for n in n_values:
    model_name = f'BMI_n{n}'
    print(f"\n{'='*60}")
    print(f" Treinando/Avaliando — BMI com n={n}")
    print(f"{'='*60}")

    local_weights = f"{MODELS_DIR}/weights-{model_name}-k{k}-n{n}-a{a}"
    local_aval    = f"{CONSTELLATION_DIR}/constellation-{model_name}-k{k}-n{n}-a{a}"
    local_ber_ser = f"{PONTOS_DIR}/BER_SER-{model_name}-k{k}-n{n}-a{a}"

    is_bit_wise = True

    with strategy.scope():
        tx = Net_Coded.encoder(k, n)
        rx = Net_Coded.decoder(k, n, a=a, bmi=True)

        model_train = End2EndSystem(k, n, tx, rx, training=True,  bit_wise=is_bit_wise)
        model_eval  = End2EndSystem(k, n, tx, rx, training=False, bit_wise=is_bit_wise)

        # Taxa de aprendizado decrescente (Cosine Decay)
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-3,
            decay_steps=NUM_TRAINING_ITER,
            alpha=1e-3
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # Treinamento (GIF e aval_training desabilitados para foco no desempenho)
        if FORCE_RETRAIN or not os.path.exists(local_weights):
            train_curriculum(model_train, snr_start=0.0, snr_end=5.0, snr_step=1.0, patience=3000,
                             optimizer=optimizer, epochs=NUM_TRAINING_ITER, batchs=BATCH_SIZE, steps_per_epoch=10,
                             local_weights=local_weights, aval_training=False, steps_for_aval=500, local_aval=local_aval,
                             generate_gif=False)

        # Recupera pesos
        model_eval = recover_weights(model_eval, local_weights)

    # Avaliação Monte Carlo
    if FORCE_RETRAIN or not os.path.exists(local_ber_ser):
        aval_model(model_eval, ebno_dbs, max_iter=150000, block_errors=500,
                   graph_mode="xla", local=local_ber_ser)

    ber_dict, ser_dict = recover_points_model(local_ber_ser)
    ber = [ber_dict[ebno] for ebno in ebno_dbs]
    ser = [ser_dict[ebno] for ebno in ebno_dbs]
    results[n] = (ber, ser)
    
    # Limpa a sessão do Keras para evitar que as variáveis do otimizador da iteração anterior
    # sejam mantidas e causem erros no `tf.function` do próximo loop.
    tf.keras.backend.clear_session()

# ============================================================================================ #
# Plot: BER e SER
# ============================================================================================ #
import matplotlib.cm as cm

# Paleta de cores para variar n (azul -> verde -> amarelo -> vermelho, etc.)
colors = cm.get_cmap('tab10')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'Desempenho Autoencoder (4,x) BMI (a={a}) variando n $\in$ [5, 9]', fontsize=16, fontweight='bold')

for ax, metric_idx, ylabel, title in [
    (axes[0], 0, 'BER', 'Bit Error Rate'),
    (axes[1], 1, 'SER', 'Symbol Error Rate'),
]:
    # Curvas dos autoencoders para cada n
    for i, n in enumerate(n_values):
        ber, ser = results[n]
        values = ber if metric_idx == 0 else ser
        ax.semilogy(ebno_dbs, values,
                    marker='o', color=colors(i), linewidth=2.0, markersize=6,
                    label=f'BMI (n={n})')

    ax.set_xlabel('Eb/N0 (dB)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f"Desempenho_Autoencoder_4_x_BMI.png"), dpi=150)
plt.show()

print(f"\nFigura salva em: {FIG_DIR}/Desempenho_Autoencoder_4_x_BMI.png")
