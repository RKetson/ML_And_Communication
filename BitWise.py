from libs.val_model import *
from libs.topology import Net_Conv_v1
from libs.model_E2E import End2EndSystem
import numpy as np
    
################
### E2E MODEL ###
################

BATCH_SIZE = 25000
NUM_TRAINING_ITERATIONS = 12500

# Parametros do sistema
k = 4
n = 7
SNRdb_train = 5.5

# Instancia as camadas do transmissor e receptor
tx = Net_Conv_v1.transmitter(k, n)
rx = Net_Conv_v1.receiver(k, n)

# Instancia o modelo fim-a-fim
model_train = End2EndSystem(k, n, tx, rx, training=True, bit_wise=True)
model = End2EndSystem(k, n, tx, rx, training=False, bit_wise=True)

# Local de dados
local_weights_4_7 = f"./Buffer/weights-{k}-{n}-neural-network-bit_wise-network"
local_aval_4_7 = f'./Buffer/constellations_energyNormalization_E2E_{k}_{n}-bit_wise-network'
local_ber_ser_4_7 = f"./Pontos/Autoencoder/AutoEncoder_{k}_{n}_ER_BitWise-network"

# Otimizador
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Treina o modelo
train(model_train, SNRdb_train, optimizer, NUM_TRAINING_ITERATIONS, BATCH_SIZE, local_weights_4_7, aval_training=True, steps_for_aval=2500, local_aval=local_aval_4_7)

# Recupera os pesos treinados
model = recover_weights(model, local_weights_4_7)

# Avalia o modelo treinado
ebno_dbs = np.arange(-4, 9, 1)

ber, ser = aval_model(model, ebno_dbs, max_iter=100000, graph_mode="xla", local=local_ber_ser_4_7)

print("BER:", ber)
print("SER:", ser)