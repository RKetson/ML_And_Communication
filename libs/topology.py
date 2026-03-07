from libs.model_E2E import EnergyNormalization
import tensorflow as tf
from collections import namedtuple
from tensorflow.keras.layers import Dense, Convolution1D, Reshape, MaxPooling1D, Flatten
from tensorflow.keras import Layer

# =============================================================================== #
"""
    Conv v1:

    Arquitetura de um transmissor e receptor completamente conectados, com camadas convolucionais no receptor.
"""


################
## TRASMITTER ##
################

class Transmitter(Layer): # Inherits from Keras Layer
    """
        Arquitetura de um transmissor completamente conectado.
    """

    def __init__(self, k, n):
        """
            Entrada: k (Dimensão de entrada), n (Dimensão de saída)
            Saída: None
        """
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
        """
            Entrada: k (Dimensão de entrada), n (Dimensão de saída)
            Saída: None
        """
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

__Net_Conv_v1 = namedtuple('Net_Conv_v1', ['transmitter', 'receiver'])

Net_Conv_v1 = __Net_Conv_v1(Transmitter, Receiver)
"""
    Net_Conv_v1:
    Arquitetura de um transmissor e receptor completamente conectados, com camadas convolucionais no receptor.
"""
Net_Conv_v1.transmitter.__doc__ = Transmitter.__doc__
Net_Conv_v1.receiver.__doc__ = Receiver.__doc__
# =============================================================================== #
