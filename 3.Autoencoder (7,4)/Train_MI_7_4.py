import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist

from libs.val_model import recover_weights, aval_model, recover_points_model, train_curriculum, train
from libs.topology import Net_Coded, Net_Full_Con
from libs.model_E2E import End2EndSystem
from libs.AFF3CT_to_points import txt_to_dict
import argparse

# ============================================================================================ #
# Argumentos de linha de comando
# ============================================================================================ #
parser = argparse.ArgumentParser(description="Treinamento MI Autoencoder 7,4")
parser.add_argument("--retrain", action="store_true", help="Força novo treinamento, sobrescrevendo resultados salvos.")
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
# Parâmetros do sistema (MI)
# ============================================================================================ #
BATCH_SIZE           = 8000
NUM_TRAINING_ITER    = 25000

k           = 4          
n           = 7          
ebno_dbs    = np.arange(-4, 8, 1)

a = 2
model_name = "MI"
is_bit_wise = False

# ============================================================================================ #
# Diretórios de saída
# ============================================================================================ #
BUFFER_DIR = "./Buffer/Fully Connected/3.Autoencoder (7,4)"
PONTOS_DIR = "./Pontos/Autoencoder/Fully Connected/3.Autoencoder (7,4)"
FIG_DIR    = "./Figures/Fully Connected/3.Autoencoder (7,4)"

os.makedirs(BUFFER_DIR, exist_ok=True)
os.makedirs(PONTOS_DIR, exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)

local_weights = f"{BUFFER_DIR}/weights-{model_name}-k{k}-n{n}-a{a}"
local_aval    = f"{BUFFER_DIR}/constellation-{model_name}-k{k}-n{n}-a{a}"
local_ber_ser = f"{PONTOS_DIR}/BER_SER-{model_name}-k{k}-n{n}-a{a}"

# ============================================================================================ #
# Treinamento e Avaliação
# ============================================================================================ #
with strategy.scope():

    tx = Net_Coded.encoder(k, n)
    rx = Net_Coded.decoder(k, n, a=a, bmi=is_bit_wise)

    model_train = End2EndSystem(k, n, tx, rx, training=True,  bit_wise=is_bit_wise)
    model_eval  = End2EndSystem(k, n, tx, rx, training=False, bit_wise=is_bit_wise)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=NUM_TRAINING_ITER,
        alpha=1e-3
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    if FORCE_RETRAIN or not os.path.exists(local_weights):
        train(model_train, snr_dB_Train=7, optimizer=optimizer, epochs=NUM_TRAINING_ITER, batchs=BATCH_SIZE,
            local_weights=local_weights, aval_training=True, steps_for_aval=2500, local_aval=local_aval)

    model_eval = recover_weights(model_eval, local_weights)

if FORCE_RETRAIN or not os.path.exists(local_ber_ser):
    aval_model(model_eval, ebno_dbs, max_iter=750000, block_errors=500, graph_mode="xla", local=local_ber_ser)
ber_dict, ser_dict = recover_points_model(local_ber_ser)
ber = [ber_dict[ebno] for ebno in ebno_dbs]
ser = [ser_dict[ebno] for ebno in ebno_dbs]

# ============================================================================================ #
# Referência: Hamming (7,4)
# ============================================================================================ #
ber_ham_ref, ser_ham_ref = txt_to_dict("./Pontos/AFF3CT/Hamming-7-4.txt")
ber_ham_ref = [ber_ham_ref.get(ebno, float('nan')) for ebno in ebno_dbs]
ser_ham_ref = [ser_ham_ref.get(ebno, float('nan')) for ebno in ebno_dbs]

ber_ham_mld, ser_ham_mld = txt_to_dict("./Pontos/AFF3CT/Hamming-7-4-MLD.txt")
ber_ham_mld = [ber_ham_mld.get(ebno, float('nan')) for ebno in ebno_dbs]
ser_ham_mld = [ser_ham_mld.get(ebno, float('nan')) for ebno in ebno_dbs]

# ============================================================================================ #
# Visualizações
# ============================================================================================ #
# 1. Curva de Desempenho (BER e SER)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'Desempenho: Autoencoder MI (7,4)', fontsize=16, fontweight='bold')

for ax, metric_idx, ylabel, title in [
    (axes[0], 0, 'BER', 'Bit Error Rate'),
    (axes[1], 1, 'SER', 'Symbol Error Rate'),
]:
    ref = ber_ham_ref if metric_idx == 0 else ser_ham_ref
    ax.semilogy(ebno_dbs, ref, 'k--', label='Hamming (7,4)')
    
    ref_mld = ber_ham_mld if metric_idx == 0 else ser_ham_mld
    ax.semilogy(ebno_dbs, ref_mld, 'g-.', label='Hamming (7,4) MLD')
    
    values = ber if metric_idx == 0 else ser
    ax.semilogy(ebno_dbs, values, 'r-o', label='Autoencoder MI (a=2)')
    
    ax.set_xlabel('Eb/N0 (dB)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", ls="--")
    ax.legend()

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/BER_SER_{model_name}_n7.png")
plt.close()

# Extração da Constelação
M = 2**k
inputs = tf.eye(M)
symbols = model_eval.transmitter(inputs).numpy()

# Gray Code
gray_indices = [0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8]
labels = [f"{i:04b}" for i in gray_indices]
symbols = symbols[gray_indices]

# 2. Matriz de Distâncias Euclidianas
dist_matrix = cdist(symbols, symbols, metric='euclidean')

plt.figure(figsize=(8, 7))
mask = np.eye(M, dtype=bool)
sns.heatmap(dist_matrix, annot=True, fmt=".1f", cmap="rocket_r", mask=mask, linewidths=0.5, linecolor='gray',
            xticklabels=labels, yticklabels=labels, cbar_kws={'fraction': 0.046, 'pad': 0.04})
plt.title('Matriz de Distâncias Euclidianas: MI (Autoencoder 7,4)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/Distances_{model_name}_n7.png")
plt.close()

# 3. t-SNE Normalizado
def normalize_tsne(coords):
    center = coords - np.mean(coords, axis=0, keepdims=True)
    energy_avg = np.mean(np.sum(np.square(center), axis=-1))
    return center / np.sqrt(energy_avg)

tsne = TSNE(n_components=2, perplexity=5, random_state=42)
pca_symbols = tsne.fit_transform(symbols)
pca_symbols = normalize_tsne(pca_symbols)

plt.figure(figsize=(7, 6))
plt.scatter(pca_symbols[:, 0], pca_symbols[:, 1], c='r', s=50)
for i, label in enumerate(labels):
    plt.annotate(label, (pca_symbols[i, 0], pca_symbols[i, 1]), textcoords="offset points", xytext=(0, 5), ha='center')
plt.grid(True)
plt.title('Projeção t-SNE 2D da Constelação 7D: MI (Symbol-wise)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
lim_max = np.max(np.abs(pca_symbols)) + 0.5
plt.xlim(-lim_max, lim_max)
plt.ylim(-lim_max, lim_max)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/tSNE_{model_name}_n7.png")
plt.close()

print(f"Treinamento e visualização MI concluídos. Arquivos gerados em {FIG_DIR}")
