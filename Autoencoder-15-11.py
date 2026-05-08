import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from libs.val_model import train, recover_weights, aval_model, recover_points_model
from libs.topology import Net_Conv_v1
from libs.model_E2E import End2EndSystem
from libs.AFF3CT_to_points import txt_to_dict

"""
    Treinamento e avaliação de um autoencoder (15,11) com abordagem bit-wise.
    Compara o desempenho com Hamming (15,11) decodificação soft e BPSK não codificado.

    Uso:
        python Autoencoder-15-11.py           # Usa pesos salvos se existirem
        python Autoencoder-15-11.py --retrain # Força novo treinamento e avaliação
"""

################
### E2E MODEL ###
################

# ============================================================================================ #
# Argumentos de linha de comando
# ============================================================================================ #
parser = argparse.ArgumentParser(description="Autoencoder (15,11) - Bit-wise")
parser.add_argument(
    "--retrain",
    action="store_true",
    help="Força um novo ciclo de treinamento e avaliação, sobrescrevendo resultados salvos."
)
args = parser.parse_args()

FORCE_RETRAIN = args.retrain

# ============================================================================================ #
# Configuração de GPUs (suporte a múltiplas GPUs via MirroredStrategy)
# ============================================================================================ #
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    strategy = tf.distribute.MirroredStrategy()
    print(f"Treinamento distribuído em {strategy.num_replicas_in_sync} GPU(s).")
else:
    strategy = tf.distribute.get_strategy()
    print("Nenhuma GPU encontrada. Usando CPU.")

# ============================================================================================ #
# Parâmetros do sistema
# ============================================================================================ #
BATCH_SIZE = 25000
NUM_TRAINING_ITERATIONS = 12500

k = 11
n = 15
SNRdb_train = 5.0
ebno_dbs = np.arange(-4, 8, 1)

# ============================================================================================ #
# Instancia o modelo
# ============================================================================================ #
with strategy.scope():
    tx = Net_Conv_v1.transmitter(k, n)
    rx = Net_Conv_v1.receiver(k, n)
    model_train = End2EndSystem(k, n, tx, rx, training=True, bit_wise=True)
    model       = End2EndSystem(k, n, tx, rx, training=False, bit_wise=True)

# Local de dados
local_weights_11_15  = f"./Buffer/Convolutional/weights-{k}-{n}-neural-network-bit_wise-network"
local_aval_11_15     = f'./Buffer/Convolutional/constellations_E2E_{k}_{n}-bit_wise-network'
local_ber_ser_11_15  = f"./Pontos/Autoencoder/Convolutional/AutoEncoder_{k}_{n}_ER_BitWise-network"

with strategy.scope():
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# ============================================================================================ #
# Treinamento
# ============================================================================================ #
if FORCE_RETRAIN or not os.path.exists(local_weights_11_15):
    train(model_train, SNRdb_train, optimizer, NUM_TRAINING_ITERATIONS, BATCH_SIZE,
          local_weights_11_15, aval_training=True, steps_for_aval=2500, local_aval=local_aval_11_15)

model = recover_weights(model, local_weights_11_15)

# ============================================================================================ #
# Avaliação
# ============================================================================================ #
if FORCE_RETRAIN or not os.path.exists(local_ber_ser_11_15):
    aval_model(model, ebno_dbs, max_iter=750000, block_errors=500, graph_mode="xla",
               local=local_ber_ser_11_15)
ber, ser = recover_points_model(local_ber_ser_11_15)
ber, ser = [ber[ebno] for ebno in ebno_dbs], [ser[ebno] for ebno in ebno_dbs]

# ============================================================================================ #
"""
    Avaliação de resultados.

    Aqui, os resultados de BER e SER do modelo utilizado são comparados com uma codificação em sistema clássico.
"""

ber_uncoded, ser_uncoded = txt_to_dict("./Pontos/AFF3CT/Uncoded-BPSK.txt")
ber_uncoded, ser_uncoded = [ber_uncoded[ebno] for ebno in ebno_dbs], [ser_uncoded[ebno] for ebno in ebno_dbs]

ber_hamming, ser_hamming = txt_to_dict("./Pontos/AFF3CT/Hamming-15-11.txt")
ber_hamming, ser_hamming = [ber_hamming[ebno] for ebno in ebno_dbs], [ser_hamming[ebno] for ebno in ebno_dbs]

fig_dir = "./Figures/Convolutional/Autoencoder-15-11"
os.makedirs(fig_dir, exist_ok=True)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.semilogy(ebno_dbs, ber, 'o-', label='Autoencoder (15,11) - Bit-wise')
plt.semilogy(ebno_dbs, ber_hamming, 's-', label='Hamming (15,11) - Soft Decision')
plt.semilogy(ebno_dbs, ber_uncoded, '^-', label='B-PSK - Não codificado')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER')
plt.title('Bit Error Rate Comparison')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.semilogy(ebno_dbs, ser, 'o-', label='Autoencoder (15,11) - Bit-wise')
plt.semilogy(ebno_dbs, ser_hamming, 's-', label='Hamming (15,11) - Soft Decision')
plt.semilogy(ebno_dbs, ser_uncoded, '^-', label='B-PSK - Não codificado')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('SER')
plt.title('Symbol Error Rate Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "BER_SER_Comparison.png"))
plt.show()
