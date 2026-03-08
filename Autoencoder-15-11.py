from libs.val_model import *
from libs.topology import Net_Conv_v1
from libs.model_E2E import End2EndSystem
from libs.AFF3CT_to_points import txt_to_dict
import numpy as np
import matplotlib.pyplot as plt
    
################
### E2E MODEL ###
################

BATCH_SIZE = 25000
NUM_TRAINING_ITERATIONS = 12500

# Parametros do sistema
k = 11
n = 15
SNRdb_train = 5.5

# Instancia as camadas do transmissor e receptor
tx = Net_Conv_v1.transmitter(k, n)
rx = Net_Conv_v1.receiver(k, n)

# Instancia o modelo fim-a-fim
model_train = End2EndSystem(k, n, tx, rx, training=True, bit_wise=True)
model = End2EndSystem(k, n, tx, rx, training=False, bit_wise=True)

# Local de dados
local_weights_11_15 = f"./Buffer/weights-{k}-{n}-neural-network-bit_wise-network"
local_aval_11_15 = f'./Buffer/constellations_energyNormalization_E2E_{k}_{n}-bit_wise-network'
local_ber_ser_11_15 = f"./Pontos/Autoencoder/AutoEncoder_{k}_{n}_ER_BitWise-network"

# Otimizador
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Treina o modelo
train(model_train, SNRdb_train, optimizer, NUM_TRAINING_ITERATIONS, BATCH_SIZE, local_weights_11_15, aval_training=True, steps_for_aval=2500, local_aval=local_aval_11_15)

# Recupera os pesos treinados
model = recover_weights(model, local_weights_11_15)

# Avalia o modelo treinado
ebno_dbs = np.arange(-4, 8, 1)

ber, ser = aval_model(model, ebno_dbs, max_iter=100000, graph_mode="xla", local=local_ber_ser_11_15)

# ============================================================================================ #
"""
    Avaliação de resultados.

    Aqui, os resultados de BER e SER do modelo utilizado são comparados com a uma codificação em sistema clássico.
"""

ber_uncoded, ser_uncoded = txt_to_dict("./Pontos/AFF3CT/Uncoded-BPSK.txt")
ber_hamming, ser_hamming = txt_to_dict("./Pontos/AFF3CT/Hamming-15-11.txt")

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.semilogy(ebno_dbs, ber, 'o-', label='Autoencoder (15,11) - Bit-wise')
plt.semilogy(ebno_dbs, ber_hamming, 's-', label='Hamming (15,11) - Soft Decision')
plt.semilogy(ebno_dbs, ber_uncoded, '^-', label='B-PSK - Não codificado')
plt.xlabel('EB/N0 (dB)')
plt.ylabel('BER')
plt.title('Bit Error Rate Comparison')
plt.legend()
plt.grid(True)


plt.subplot(1, 2, 2)
plt.semilogy(ebno_dbs, ser, 'o-', label='Autoencoder (15,11) - Bit-wise')
plt.semilogy(ebno_dbs, ser_hamming, 's-', label='Hamming (15,11) - Soft Decision')
plt.semilogy(ebno_dbs, ser_uncoded, '^-', label='B-PSK - Não codificado')
plt.xlabel('EB/N0 (dB)')
plt.ylabel('SER')
plt.title('Symbol Error Rate Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
