import libs.tf_config
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

class EnergyNormalization(tf.keras.Layer):
    """
        Camada personalizada para normalização por energia.
    """
    def __init__(self, **kwargs):
        super(EnergyNormalization, self).__init__(**kwargs)
    
    def call(self, input):
        x = input
        center = x - tf.reduce_mean(x, axis=0, keepdims=True)


        energy_avg = tf.reduce_mean(tf.reduce_sum(tf.square(center), axis=-1, keepdims=True), axis=0, keepdims=True)

        energy_sqrt = tf.sqrt(energy_avg)

        x_norm = center / energy_sqrt

        return x_norm
    
class End2EndSystem(tf.keras.Model): # Inherits from Keras Model
    """
      Modelo criado para simular um sistem Fim-a-Fim com camadas treináveis.
    """

    def __init__(self, k, n, tx, rx, training=False, bit_wise=False):
        """
        Entradas:
          k: Quantidade de bits de entrada
          n: Quantidade de bits de saída
          tx: Camada do transmissor (baseada em symbol-wise ou bit-wise).
          rx: Camada do receptor (baseada em symbol-wise ou bit-wise).
          training: True se o modelo for treinável, Falso caso não.
          bit_wise: True se o modelo for bit-wise, False caso seja symbol-wise.
        """

        super().__init__() # Must call the Keras model initializer

        self.k = k
        self.n = n
        self.M = 2**k
        self.coderate = k/n

        self.rng = tf.random.get_global_generator()
        self.transmitter = tx
        self.receiver = rx

        if bit_wise:
          self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        else:
          self.bce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        self.training = training
        self.bit_wise = bit_wise

    def bits_to_indices(self, bits):
      """
        Converter mapeia os bits de entrada em índice seguindo uma contagem binária.
      """

      # Reverse é utilizado para que o bit mais significativo esteja a esquerda
      binary_patches = tf.reverse(tf.reshape(bits, (-1, self.k)), axis=[-1])
      # Converter cada patch para decimal
      return tf.reduce_sum(binary_patches * (2 ** tf.range(self.k, dtype=tf.int32)), axis=-1)

    def convert_symbol_probs_to_bit_probs_graph_compatible(self, symbol_probs):
        """
        Converte um tensor de probabilidades de símbolos (2^k classes)
        em um tensor de probabilidades de bits individuais, otimizado para grafos.

        Args:
            symbol_probs (tf.Tensor): Tensor de probabilidades de símbolos
                                      com formato [batch_size, 2^k_bits].
                                      Cada linha deve somar 1.
        Returns:
            tf.Tensor: Tensor de probabilidades de bits com formato [batch_size, k_bits].
                      Cada elemento [b, i] representa P(i-ésimo bit é 1) para a amostra 'b'.
        """
        num_symbols = 2**self.k

        symbols_indices = tf.range(num_symbols, dtype=tf.int32)

        powers_of_2 = tf.pow(2, tf.range(self.k - 1, -1, -1)) # Shape: [k_bits]

        symbols_indices_expanded = tf.expand_dims(symbols_indices, axis=1)

        powers_of_2_expanded = tf.expand_dims(powers_of_2, axis=0)

        bit_flags = tf.bitwise.bitwise_and(symbols_indices_expanded, powers_of_2_expanded)

        bit_map = tf.cast(tf.math.greater(bit_flags, 0), dtype=tf.float32)

        bit_probabilities = tf.matmul(symbol_probs, bit_map)

        return bit_probabilities

    def indices_to_bits(self, indices):
      """
        Mapeia os índices decimais em bits, seguindo a conversão binária do índice.
      """

      powers = tf.range(self.k-1, -1, -1)

      # Expande as dimensões do tensor para que possamos fazer a operação de bitwise
      tensor_expanded = tf.expand_dims(indices, -1)
      tensor_expanded = tf.cast(tensor_expanded, tf.int32)

      # Faz a operação de bitwise e extrai os bits
      binary_tensor = tf.bitwise.bitwise_and(tf.bitwise.right_shift(tensor_expanded, powers), 1)

      # Redimensiona o tensor para o formato desejado
      return tf.reshape(binary_tensor, (-1, self.k))

    def points_Constellation(self, bits=None):
      """
        Retorna uma quantidade 'samples' da saída do transmissor.
        
        Obs.: Ruído não é incluído.
      """
      if bits is None:
        numeros = tf.range(2**self.k, dtype=tf.int32)

        bits = tf.reverse(
            tf.bitwise.right_shift(numeros[:, tf.newaxis], tf.range(self.k, dtype=tf.int32)) & 1,
            axis=[1]
        )

        if self.bit_wise:
          z = self.transmitter(bits)
        else:
          indices = self.bits_to_indices(bits)
          one_hot = tf.one_hot(indices, depth=self.M)
          z = self.transmitter(one_hot)

        return bits, z
      
      else:

        if self.bit_wise:
          z = self.transmitter(bits)
        else:
          indices = self.bits_to_indices(bits)
          one_hot = tf.one_hot(indices, depth=self.M)
          z = self.transmitter(one_hot)
          
        return z
      
    def calcular_distancias_ordem_n(self, z):
        from scipy.spatial.distance import pdist, squareform

        if isinstance(z, tf.Tensor):
            z = z.numpy()
        return squareform(pdist(z, metric='euclidean'))    

    def plot_transmitter(self, vizinhos=None):
       
      fig, ax = plt.subplots(figsize=(7, 7))

      bits, z = self.points_Constellation()
      labels_binarios_str = ["".join(str(int(bit)) for bit in row) for row in bits]

      if vizinhos is not None:
          dist_eucled = self.calcular_distancias_ordem_n(z)

      def plot(z, label=True):

        for i, point in enumerate(z):
          x, y = point

          ax.plot(x, y, 'bo')
          if label:
              ax.annotate(labels_binarios_str[i], (x, y),
                          textcoords="offset points", xytext=(5,5), ha='center', fontsize=9)         

      if z.shape[-1] == 2:
          plot(z=z)

      else:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import MinMaxScaler

        tsne = TSNE(n_components=2, perplexity=2**self.k - 1, random_state=42)
        z_transform = tsne.fit_transform(z)

        scaler = MinMaxScaler()
        z_transform = scaler.fit_transform(z_transform)

        plot(z=z_transform, label=False)

      if vizinhos is not None:
          
          # Armazenar pares vizinhos + distância para exibição em tabela
          vizinhos_por_ponto = {}

          if z.shape[-1] == 2:
              z_plot = z
          else:
              z_plot = z_transform

          # Para cada ponto, encontrar os k vizinhos mais próximos (ignorando ele mesmo)
          N = len(z)
          for i in range(N):
              distancias = dist_eucled[i]
              indices_vizinhos = np.argsort(distancias)[1:vizinhos + 1]  # Ignora o próprio ponto [0]
              vizinhos_por_ponto[i] = []

              #x1, y1 = z_plot[i]
              print("#"*40)
              for j in indices_vizinhos:
                  #x2, y2 = z_plot[j]
                  dist = dist_eucled[i, j]

                  print(f"Distância entre {labels_binarios_str[i]} e {labels_binarios_str[j]}: {dist:.4f}")

                  #ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.3, linestyle='--')
                  #xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
                  #ax.text(xm, ym, f"{dist:.2f}", fontsize=7, color='red', ha='center')

      ax.set_title(f'Transmissor')
      ax.set_xlabel('Parte Real')
      ax.set_ylabel('Parte Imaginária')
      ax.grid(True)
      ax.set_aspect('equal', adjustable='box')

      plt.tight_layout()
      plt.show()

    def add_GaussianNoise(self, y, ebno_db):
      """
        Adiciona um ruído de 'ebno_db' aos símbolos transmitidos.
      """

      # Calcular o valor do SNR linear
      snr_linear = tf.pow(10.0, ebno_db / 10.0)

      noise_psd = 1.0 / (self.coderate * snr_linear * self.n)

      # Gerar ruído gaussiano ajustado para a dimensionalidade
      noise = tf.random.normal(shape=tf.shape(y), 
                              mean=0.0,
                              stddev=tf.sqrt(noise_psd),
                              dtype=y.dtype)
      
      # Adicionar o ruído ao sinal codificado
      noisy_signal = tf.add(y, noise)

      return noisy_signal

    @tf.function(jit_compile=True) # Enable graph execution to speed things up
    def __call__(self, batch_size, ebno_db):
        """
          Realiza uma quantidade 'batch_size' de transmissões pelo canal com um SNR de 'ebno_db', demodula e decodifica
          usando um receptor baseado em symbol-wise ou bit-wise e usa a função binária de entropia cruzada como função de perda (esparça para symbol-wise).
        """

        bits = self.rng.uniform([batch_size, self.k], 0, 2, tf.int32)
        
        if self.bit_wise:
            z = self.transmitter(bits)
        else:
            indices = self.bits_to_indices(bits)
            one_hot = tf.one_hot(indices, depth=self.M)
            z = self.transmitter(one_hot)

        y = self.add_GaussianNoise(z, ebno_db)
        
        recev = self.receiver(y)

        if self.training:
            if self.bit_wise:
                loss = self.bce(bits, recev)
            else:
                loss = self.bce(one_hot, recev)
            return loss
        else:
            if self.bit_wise:
              bits_hat = tf.math.greater_equal(recev, 0.5)
              bits_hat = tf.cast(bits_hat, dtype=tf.int32)
              return bits, bits_hat
            else:
              return bits, self.indices_to_bits(tf.math.argmax(recev, axis=-1))