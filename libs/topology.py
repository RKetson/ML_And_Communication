from libs.model_E2E import EnergyNormalization
import tensorflow as tf
from collections import namedtuple
from tensorflow.keras.layers import Dense, Convolution1D, Reshape, MaxPooling1D, Flatten
from tensorflow.keras import Layer

"""
    Fully connected
"""
################
## TRASMITTER ##
################

class Transmitter_FL(Layer):
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

class Receiver_FL(Layer): # Inherits from Keras Layer

    def __init__(self, k, n, bit_wise=True):
        """
            Entrada: k (Dimensão de entrada), n (Dimensão de saída)
            Saída: None
        """
        super().__init__()
        self.bit_wise = bit_wise
        M = 2**k

        # The two dense layers that form the custom trainable neural network-based demapper
        self.dense_0 = Dense(M, 'relu')

        if self.bit_wise:
            self.dense_1 = Dense(M, 'relu')
            self.dense_2 = Dense(k, 'sigmoid')
        else:
            self.dense_2 = Dense(M, 'softmax')

    def call(self, y):
        z = self.dense_0(y)
        if self.bit_wise:
            z = self.dense_1(z)
        z = self.dense_2(z)
        return z

__Net_Full_Con = namedtuple('Net_Full_Con', ['transmitter', 'receiver'])

Net_Full_Con = __Net_Full_Con(Transmitter_FL, Receiver_FL)
"""
    Net_Full_Con:
    Arquitetura de um transmissor e receptor completamente conectados.
"""
Net_Full_Con.transmitter.__doc__ = Transmitter_FL.__doc__
Net_Full_Con.receiver.__doc__ = Receiver_FL.__doc__
# =============================================================================== #



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

    def __init__(self, k, n, bit_wise=True):
        """
            Entrada: k (Dimensão de entrada), n (Dimensão de saída)
            Saída: None
        """
        super().__init__()
        self.bit_wise = bit_wise

        self.reshape = Reshape((n, 1))
        self.conv1d1 = Convolution1D(128, 2, strides=1, padding='valid', activation='relu')
        self.maxpool1 = MaxPooling1D(2)
        self.conv1d2 = Convolution1D(64, 2, strides=1, padding='valid', activation='relu')
        self.maxpool2 = MaxPooling1D(2)
        self.flatten = Flatten()
        if self.bit_wise:
            self.dense_0 = Dense(2**k, 'relu')
            self.dense_1 = Dense(k, 'sigmoid')
        else:
            self.dense_1 = Dense(2**k, 'softmax')

    def call(self, y):
        """
            Entrada: Tensor (, n)
            Saída: Tensor (, M) com logits ou probabilidades
        """
        z = self.reshape(y)
        z = self.conv1d1(z)
        z = self.maxpool1(z)
        z = self.conv1d2(z)
        z = self.maxpool2(z)
        z = self.flatten(z)
        if self.bit_wise:
            z = self.dense_0(z)
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



# =============================================================================== #
"""
    Net_BMI:

    Arquitetura para maximização da Informação Mútua de Bit (BMI).

    Transmissor:
        Recebe uma representação one-hot de tamanho M=2^k e mapeia para um ponto
        2D (I, Q) através de uma camada Dense sem bias. Os pesos desta camada
        representam diretamente as coordenadas da constelação treinável.

    Receptor:
        Recebe o sinal ruidoso 2D (y_I, y_Q) e produz k logits (equivalentes a LLRs)
        para cada bit. A capacidade do receptor é controlada pelo parâmetro `a`:
        o número de neurônios da camada oculta é 2^(k+a).
"""


################
## TRANSMITTER ##
################

class Transmitter_BMI(Layer):
    """
    Transmissor para aprendizado de constelação (BMI).

    Mapeia um vetor one-hot de tamanho M=2^k para um símbolo 2D (I, Q).
    A camada Dense sem bias age como uma tabela de constelação treinável:
    cada linha da matriz de pesos W ∈ R^(M×2) representa as coordenadas
    (I, Q) de um símbolo.

    Entradas:
        k: Número de bits de informação por símbolo (M = 2^k símbolos).
    """

    def __init__(self, k):
        super().__init__()
        M = 2**k
        # Camada de constelação: one-hot(M) → (I, Q)
        # use_bias=False garante que o ponto de origem (0,0) não seja forçado como símbolo
        self.constellation = Dense(2, activation='linear', use_bias=False)
        self.normalization = EnergyNormalization()

    def call(self, one_hot):
        """
        Entrada: Tensor one-hot de tamanho (batch, M).
        Saída:   Tensor de símbolos normalizados de tamanho (batch, 2).
        """
        z = self.constellation(one_hot)
        z = self.normalization(z)
        return z


################
### RECEIVER ###
################

class Receiver_BMI(Layer):
    """
    Receptor para demodulação bit-wise (BMI).

    Recebe o sinal ruidoso 2D e produz k logits (LLRs não normalizados),
    um por bit de informação. A saída é linear (sem sigmoid), compatível
    com BinaryCrossentropy(from_logits=True) durante o treinamento e
    com limiar em 0 durante a inferência.

    O parâmetro `a` controla a capacidade da camada oculta:
        neurônios_ocultos = 2^(k + a)

    Entradas:
        k: Número de bits de informação por símbolo.
        a: Expoente de capacidade (default=0). a=0 → 2^k neurônios.
    """

    def __init__(self, k, a=0):
        super().__init__()
        hidden_size = 2 ** (k + a)
        self.dense_hidden = Dense(hidden_size, activation='relu')
        self.dense_output = Dense(k, activation='linear')   # logits / LLRs

    def call(self, y):
        """
        Entrada: Tensor do sinal recebido de tamanho (batch, 2).
        Saída:   Tensor de logits de tamanho (batch, k).
        """
        z = self.dense_hidden(y)
        z = self.dense_output(z)
        return z


__Net_BMI = namedtuple('Net_BMI', ['transmitter', 'receiver'])

Net_BMI = __Net_BMI(Transmitter_BMI, Receiver_BMI)
"""
    Net_BMI:
    Transmissor com constelação treinável 2D + receptor bit-wise com saída em logits.
    Use com End2EndSystem(k, n=2, tx, rx, bmi=True).
"""
Net_BMI.transmitter.__doc__ = Transmitter_BMI.__doc__
Net_BMI.receiver.__doc__ = Receiver_BMI.__doc__
# =============================================================================== #
