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
    Experimento: Comparação entre Maximização da Informação Mútua (MI) e
    Maximização da Informação Mútua de Bit (BMI), utilizando a melhor
    capacidade (parâmetro 'a') encontrada para cada rede.

    Uso:
        python Compare_MI_vs_BMI_coded.py           # Carrega resultados salvos
        python Compare_MI_vs_BMI_coded.py --retrain # Força novo treinamento
"""

# ============================================================================================ #
# Argumentos de linha de comando
# ============================================================================================ #
parser = argparse.ArgumentParser(description="Comparação MI vs BMI Autoencoder")
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
BATCH_SIZE           = 25000
NUM_TRAINING_ITER    = 8000

k           = 11         # Bits de informação por símbolo
n           = 15         # Dimensões reais do símbolo transmitido (I e Q)
SNRdb_train = 4.0        # SNR de treinamento (dB)
ebno_dbs    = np.arange(-4, 7, 1)

# Defina aqui os melhores valores de 'a' encontrados nos scripts Compare_BMI_FL.py e Compare_MI_FL.py
# (Exemplo: se descobrir que a=2 é melhor para BMI e a=3 para MI, modifique estas variáveis)
BEST_A_BMI = 4
BEST_A_MI  = 2

# ============================================================================================ #
# Diretórios de saída
# ============================================================================================ #
BUFFER_DIR = "./Buffer/Fully Connected/MI_vs_BMI"
PONTOS_DIR = "./Pontos/Autoencoder/Fully Connected/MI_vs_BMI"
FIG_DIR    = "./Figures/Fully Connected/Autoencoder-MI_vs_BMI"

os.makedirs(BUFFER_DIR, exist_ok=True)
os.makedirs(PONTOS_DIR, exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)

# ============================================================================================ #
# Dicionário de modelos para iteração
# ============================================================================================ #
models_info = {
    'BMI': {
        'net': Net_Coded,
        'bit_wise': True,
        'a': BEST_A_BMI,
        'label': f'Bit-wise (a={BEST_A_BMI})'
    },
    'MI': {
        'net': Net_Coded,
        'bit_wise': False,
        'a': BEST_A_MI,
        'label': f'Symbol-wise (a={BEST_A_MI})'
    }
}

results = {}  # {nome_modelo: (ber_list, ser_list)}

for model_name, info in models_info.items():
    a = info['a']
    print(f"\n{'='*60}")
    print(f" Treinando/Avaliando — {info['label']}")
    print(f"{'='*60}")

    local_weights = f"{BUFFER_DIR}/weights-{model_name}-k{k}-n{n}-a{a}"
    local_aval    = f"{BUFFER_DIR}/constellation-{model_name}-k{k}-n{n}-a{a}"
    local_ber_ser = f"{PONTOS_DIR}/BER_SER-{model_name}-k{k}-n{n}-a{a}"

    net_topology = info['net']
    is_bit_wise = info['bit_wise']

    with strategy.scope():
        if is_bit_wise:
            tx = net_topology.encoder(k, n)
            rx = net_topology.decoder(k, n, a=a, bmi=True)
        else:
            tx = net_topology.encoder(k, n)
            rx = net_topology.decoder(k, n, a=a, bmi=False)

        model_train = End2EndSystem(k, n, tx, rx, training=True,  bit_wise=is_bit_wise)
        model_eval  = End2EndSystem(k, n, tx, rx, training=False, bit_wise=is_bit_wise)

        # Taxa de aprendizado decrescente (Cosine Decay)
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-3,
            decay_steps=NUM_TRAINING_ITER,
            alpha=1e-5   # LR mínima ao final do treino
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # Treinamento
        if FORCE_RETRAIN or not os.path.exists(local_weights):
            train(model_train, SNRdb_train, optimizer, NUM_TRAINING_ITER, BATCH_SIZE,
                local_weights, aval_training=True, steps_for_aval=2500, local_aval=local_aval)

        # Recupera pesos
        model_eval = recover_weights(model_eval, local_weights)

    # Avaliação Monte Carlo
    if FORCE_RETRAIN or not os.path.exists(local_ber_ser):
        aval_model(model_eval, ebno_dbs, max_iter=750000, block_errors=500,
                   graph_mode="xla", local=local_ber_ser)

    ber_dict, ser_dict = recover_points_model(local_ber_ser)
    ber = [ber_dict[ebno] for ebno in ebno_dbs]
    ser = [ser_dict[ebno] for ebno in ebno_dbs]
    results[model_name] = (ber, ser)

# ============================================================================================ #
# Referência: Hamming (15,11)
# ============================================================================================ #
ber_ham_ref, ser_ham_ref = txt_to_dict("./Pontos/AFF3CT/Hamming-15-11.txt")
ber_ham_ref = [ber_ham_ref.get(ebno, float('nan')) for ebno in ebno_dbs]
ser_ham_ref = [ser_ham_ref.get(ebno, float('nan')) for ebno in ebno_dbs]

# ============================================================================================ #
# Plot: BER e SER
# ============================================================================================ #
markers = {'BMI': 'o', 'MI': 's'}
colors  = {'BMI': '#e63946', 'MI': '#457b9d'}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'Comparação de Desempenho: Autoencoder (15,11) MI vs BMI', fontsize=16, fontweight='bold')

for ax, metric_idx, ylabel, title in [
    (axes[0], 0, 'BER', 'Bit Error Rate'),
    (axes[1], 1, 'SER', 'Symbol Error Rate'),
]:
    # Curvas dos autoencoders
    for model_name, info in models_info.items():
        ber, ser = results[model_name]
        values = ber if metric_idx == 0 else ser
        ax.semilogy(ebno_dbs, values,
                    marker=markers[model_name], color=colors[model_name], linewidth=1.8, markersize=6,
                    label=info['label'])

    # Referência: Hamming(15,11) não codificado simulado via AFF3CT v4.3.1
    ref = ber_ham_ref if metric_idx == 0 else ser_ham_ref
    ax.semilogy(ebno_dbs, ref, 'k-', linewidth=2.0, marker='x', markersize=7,
                label='Hamming(15,11) com Soft Decision')

    ax.set_xlabel('Eb/N0 (dB)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f"MI_vs_BMI_Best_Capacity_comparison.png"), dpi=150)
plt.show()

print(f"\nFigura salva em: {FIG_DIR}/MI_vs_BMI_Best_Capacity_comparison.png")
