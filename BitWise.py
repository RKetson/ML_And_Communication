from libs.AFF3CT_to_points import txt_to_dict
from libs.model_E2E import *
from libs.val_model import *
import numpy as np

################
## TRASMITTER ##
################

class Transmitter(Layer): # Inherits from Keras Layer
    """
        Arquitetura de um transmissor completamente conectado.
    """

    def __init__(self, k, n):
        super().__init__()

        self.dense_1 = Dense(2**k, 'relu')
        self.dense_2 = Dense(n, 'linear')
        self.dense_3 = EnergyNormalization()

    def call(self, bits):
        """
            Entrada: Tensor de bits de tamanho (, k)
            Saída: Tensor no formato (, n)
        """

        nn_input = tf.cast(bits, dtype=tf.float32)

        z = self.dense_1(nn_input)
        z = self.dense_2(z)
        z = self.dense_3(z)

        return z
    

################
### RECEIVER ###
################

class Receiver(Layer): # Inherits from Keras Layer
    """
        Arquitetura de um receptor completamente conectado com saída softmax.
    """

    def __init__(self, k, n):
        super().__init__()

        self.reshape = Reshape((n, 1))
        self.conv1d1 = Convolution1D(128, 2, strides=1, padding='valid', activation='relu')
        self.maxpool1 = MaxPooling1D(2)
        self.conv1d2 = Convolution1D(64, 2, strides=1, padding='valid', activation='relu')
        self.maxpool2 = MaxPooling1D(2)
        self.flatten = Flatten()
        self.dense_1 = Dense(k, 'sigmoid')

    def call(self, y):
        """
            Entrada: Tensor (, n)
            Saída: Tensor (, M)

            Obs.: Saídas no formato de logits
        """

        nn_input = y
        z = self.reshape(nn_input)
        z = self.conv1d1(z)
        z = self.maxpool1(z)
        z = self.conv1d2(z)
        z = self.maxpool2(z)
        z = self.flatten(z)
        z = self.dense_1(z)
        
        return z
    

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
tx = Transmitter(k, n)
rx = Receiver(k, n)

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