import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from libs.val_model import train, recover_weights, aval_model, recover_points_model
from libs.topology import Net_Full_Con
from libs.model_E2E import End2EndSystem
from libs.AFF3CT_to_points import txt_to_dict

"""
    Este arquivo é responsável por comparar o desempenho de um modelo treinado com abordagem bit-wise
    e outro com abordagem symbol-wise.
    Ele treina ambos os modelos, avalia seu desempenho em termos de BER e SER, e salva os resultados
    para posterior análise.

    Uso:
        python Compare_Bit_and_Symbol_Wise_FL.py           # Usa pesos salvos se existirem
        python Compare_Bit_and_Symbol_Wise_FL.py --retrain # Força novo treinamento e avaliação
"""

# ============================================================================================ #
# Argumentos de linha de comando
# ============================================================================================ #
parser = argparse.ArgumentParser(description="Comparação Bit-wise vs Symbol-wise Autoencoder (7,4)")
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
    # Habilita crescimento de memória para evitar alocação total no início
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    strategy = tf.distribute.MirroredStrategy()
    print(f"Treinamento distribuído em {strategy.num_replicas_in_sync} GPU(s).")
else:
    strategy = tf.distribute.get_strategy()  # Estratégia padrão (CPU ou 1 GPU)
    print("Nenhuma GPU encontrada. Usando CPU.")

# ============================================================================================ #
# Parâmetros do sistema
# ============================================================================================ #
BATCH_SIZE = 3000
NUM_TRAINING_ITERATIONS = 25000

k = 4
n = 7
SNRdb_train = 7.0
ebno_dbs = np.arange(-4, 8, 1)

# ============================================================================================ #
"""
    Seção para o modelo bit-wise.
"""

#################
### E2E MODEL ###
#################

# Instancia os modelos dentro do escopo da estratégia de distribuição
with strategy.scope():
    bit_tx = Net_Full_Con.transmitter(k, n)
    bit_rx = Net_Full_Con.receiver(k, n)
    bit_model_train = End2EndSystem(k, n, bit_tx, bit_rx, training=True, bit_wise=True)
    bit_model       = End2EndSystem(k, n, bit_tx, bit_rx, training=False, bit_wise=True)

# Local de dados
bit_local_weights_4_7  = f"./Buffer/Fully Connected/weights-{k}-{n}-neural-network-bit_wise-network_v2"
bit_local_aval_4_7     = f'./Buffer/Fully Connected/constellations_energyNormalization_E2E_{k}_{n}-bit_wise-network_v2'
bit_local_ber_ser_4_7  = f"./Pontos/Autoencoder/Fully Connected/AutoEncoder_{k}_{n}_ER_BitWise-network_v2"

# Otimizador (instanciado dentro do scope para distribuição correta)
with strategy.scope():
    bit_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Treina o modelo (apenas se necessário ou forçado por --retrain)
if FORCE_RETRAIN or not os.path.exists(bit_local_weights_4_7):
    train(bit_model_train, SNRdb_train, bit_optimizer, NUM_TRAINING_ITERATIONS, BATCH_SIZE,
          bit_local_weights_4_7, aval_training=True, steps_for_aval=2500, local_aval=bit_local_aval_4_7)

# Recupera os pesos treinados
bit_model = recover_weights(bit_model, bit_local_weights_4_7)

# Avalia o modelo treinado
if FORCE_RETRAIN or not os.path.exists(bit_local_ber_ser_4_7):
    aval_model(bit_model, ebno_dbs, max_iter=750000, block_errors=500, graph_mode="xla",
               local=bit_local_ber_ser_4_7)
bit_ber, bit_ser = recover_points_model(bit_local_ber_ser_4_7)
bit_ber, bit_ser = [bit_ber[ebno] for ebno in ebno_dbs], [bit_ser[ebno] for ebno in ebno_dbs]

# ============================================================================================ #
"""
    Seção para o modelo symbol-wise.
"""

#################
### E2E MODEL ###
#################

# Instancia os modelos dentro do escopo da estratégia de distribuição
with strategy.scope():
    symbol_tx = Net_Full_Con.transmitter(k, n)
    symbol_rx = Net_Full_Con.receiver(k, n, bit_wise=False)
    symbol_model_train = End2EndSystem(k, n, symbol_tx, symbol_rx, training=True, bit_wise=False)
    symbol_model       = End2EndSystem(k, n, symbol_tx, symbol_rx, training=False, bit_wise=False)

# Local de dados
symbol_local_weights_4_7  = f"./Buffer/Fully Connected/weights-{k}-{n}-neural-network-symbol_wise-network_v2"
symbol_local_aval_4_7     = f'./Buffer/Fully Connected/constellations_energyNormalization_E2E_{k}_{n}-symbol_wise-network_v2'
symbol_local_ber_ser_4_7  = f"./Pontos/Autoencoder/Fully Connected/AutoEncoder_{k}_{n}_ER_SymbolWise-network_v2"

# Otimizador
with strategy.scope():
    symbol_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Treina o modelo (apenas se necessário ou forçado por --retrain)
if FORCE_RETRAIN or not os.path.exists(symbol_local_weights_4_7):
    train(symbol_model_train, SNRdb_train, symbol_optimizer, NUM_TRAINING_ITERATIONS, BATCH_SIZE,
          symbol_local_weights_4_7, aval_training=True, steps_for_aval=2500, local_aval=symbol_local_aval_4_7)

# Recupera os pesos treinados
symbol_model = recover_weights(symbol_model, symbol_local_weights_4_7)

# Avalia o modelo treinado
if FORCE_RETRAIN or not os.path.exists(symbol_local_ber_ser_4_7):
    aval_model(symbol_model, ebno_dbs, max_iter=750000, block_errors=500, graph_mode="xla",
               local=symbol_local_ber_ser_4_7)
symbol_ber, symbol_ser = recover_points_model(symbol_local_ber_ser_4_7)
symbol_ber, symbol_ser = [symbol_ber[ebno] for ebno in ebno_dbs], [symbol_ser[ebno] for ebno in ebno_dbs]

# ============================================================================================ #
"""
    Comparação dos resultados.

    Aqui, os resultados de BER e SER dos modelos bit-wise e symbol-wise são comparados e visualizados.
"""

ber, ser = txt_to_dict("./Pontos/AFF3CT/Hamming-7-4-MLD.txt")
ber, ser = [ber[ebno] for ebno in ebno_dbs], [ser[ebno] for ebno in ebno_dbs]

ber_uncoded, ser_uncoded = txt_to_dict("./Pontos/AFF3CT/Uncoded-BPSK.txt")
ber_uncoded, ser_uncoded = [ber_uncoded[ebno] for ebno in ebno_dbs], [ser_uncoded[ebno] for ebno in ebno_dbs]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.semilogy(ebno_dbs, bit_ber, 'o-', label='Autoencoder (7,4) - Bit-wise')
plt.semilogy(ebno_dbs, symbol_ber, 's-', label='Autoencoder (7,4) - Symbol-wise')
plt.semilogy(ebno_dbs, ber, '^-', label='Hamming (7,4) - Soft Decision')
plt.semilogy(ebno_dbs, ber_uncoded, '--', label='B-PSK - Não codificado')
plt.xlabel('EB/N0 (dB)')
plt.ylabel('BER')
plt.title('Bit Error Rate Comparison')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.semilogy(ebno_dbs, bit_ser, 'o-', label='Autoencoder (7,4) - Bit-wise')
plt.semilogy(ebno_dbs, symbol_ser, 's-', label='Autoencoder (7,4) - Symbol-wise')
plt.semilogy(ebno_dbs, ser, '^-', label='Hamming (7,4) - Soft Decision')
plt.semilogy(ebno_dbs, ser_uncoded, '--', label='B-PSK - Não codificado')
plt.xlabel('EB/N0 (dB)')
plt.ylabel('SER')
plt.title('Symbol Error Rate Comparison')
plt.legend()
plt.grid(True)

fig_dir = "./Figures/Fully Connected/Autoencoder-7-4"
os.makedirs(fig_dir, exist_ok=True)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "Bit-Wise_Symbol-Wise_Comparison.png"))
plt.show()
