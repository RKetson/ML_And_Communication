import libs.tf_config
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

class EnergyNormalization(tf.keras.Layer):
    """
    Camada personalizada para normalização por energia.

    Garante que E_s = 1, ou seja, a energia média por símbolo transmitido é unitária.
    Isso é necessário para que a parametrização por Eb/N0 seja correta e comparável
    entre diferentes modelos e runs.

    NOTA TEÓRICA:
        A versão anterior subtraía a média do batch (centralização) antes de normalizar.
        Isso é problemático por dois motivos:
          1. Introduce bias no gradiente: a centralização é uma função do batch, tornando
             o ponto da constelação dependente dos outros símbolos do batch atual.
          2. Viola a constraint de energia: E[||x||^2] = 1 deve valer sobre toda a
             constelação, não sobre um batch específico. A formulação correta normaliza
             pelo módulo médio de energia sem remover a média DC.
    """
    def __init__(self, **kwargs):
        super(EnergyNormalization, self).__init__(**kwargs)

    def call(self, input):
        # Energia média por símbolo: média de ||x_i||^2 sobre o batch
        # Shape: escalar (média sobre batch e dimensões do símbolo)
        energy_avg = tf.reduce_mean(tf.reduce_sum(tf.square(input), axis=-1))

        # Normaliza para que E_s = 1 (energia média unitária)
        x_norm = input / tf.sqrt(energy_avg)

        return x_norm
    
class End2EndSystem(tf.keras.Model): # Inherits from Keras Model
    """
      Modelo criado para simular um sistem Fim-a-Fim com camadas treináveis.
    """

    def __init__(self, k, n, tx, rx, training=False, bit_wise=False, bmi=False):
        """
        Entradas:
          k:        Quantidade de bits de entrada por símbolo.
          n:        Número de dimensões reais do símbolo transmitido.
          tx:       Camada do transmissor.
          rx:       Camada do receptor.
          training: True se o modelo for usado para treino.
          bit_wise: True para abordagem bit-wise (transmissor recebe bits, receptor sigmoid).
          bmi:      True para modo BMI — transmissor recebe one-hot, receptor produz
                    logits (LLRs) com BCE from_logits=True. Incompatível com bit_wise=True.
        """

        super().__init__()

        self.k = k
        self.n = n
        self.M = 2**k
        self.coderate = k/n

        self.rng = tf.random.get_global_generator()
        self.transmitter = tx
        self.receiver = rx
        self.bmi = bmi

        if bmi:
          # Logits como saída: BCE com from_logits=True é numericamente mais estável
          self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        elif bit_wise:
          self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        else:
          self.bce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        self.is_training = training
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
      Adiciona ruído AWGN gaussiano ao sinal transmitido, parametrizado por Eb/N0.

      DERIVAÇÃO DA VARIÂNCIA DO RUÍDO:
          - Eb/N0 (linear) = ebno_linear
          - Taxa de código: R = k/n
          - Energia por símbolo: E_s = 1 (garantido pela EnergyNormalization)
          - Dimensões reais por símbolo: n (cada saída do transmissor é real)
          - Relação entre Es/N0 e Eb/N0: Es/N0 = Eb/N0 * R * n
            (pois E_s = E_b * k bits por símbolo, e são n dimensões reais)
          - Variância do ruído por dimensão: sigma^2 = N0/2 = E_s / (2 * Es/N0)
            => sigma^2 = 1 / (2 * ebno_linear * R * n)

      CORREÇÃO EM RELAÇÃO À VERSÃO ANTERIOR:
          A versão anterior usava noise_psd = 1/(R * SNR * n), que corresponde
          à variância total, não por dimensão. Como o tf.random.normal gera uma
          amostra independente por dimensão, a variância já é por componente,
          portanto o fator 2 (que divide N0 em cada dimensão I e Q do canal complexo)
          deve ser aplicado. Para canal puramente real, sigma^2 = N0/2.
      """
      ebno_linear = tf.pow(10.0, ebno_db / 10.0)

      # Variância do ruído por dimensão real: sigma^2 = N0/2
      # N0 = E_s / (Es/N0) = 1 / (ebno_linear * coderate * n)
      # sigma^2 = N0/2 = 1 / (2 * ebno_linear * coderate * n)
      noise_variance = 1.0 / (2.0 * self.coderate * ebno_linear * self.n)

      noise = tf.random.normal(
          shape=tf.shape(y),
          mean=0.0,
          stddev=tf.sqrt(noise_variance),
          dtype=y.dtype
      )

      return tf.add(y, noise)

    @tf.function(jit_compile=True)
    def __call__(self, batch_size, ebno_db):
        """
        Realiza 'batch_size' transmissões com SNR 'ebno_db' e retorna a perda
        (modo treino) ou o par (bits, bits_hat) (modo inferência).

        Modos suportados:
          - bmi=True:      Transmissor one-hot, receptor logits, BCE from_logits=True.
                           Inferência: limiar dos logits em 0 (equiv. sigmoid ≥ 0.5).
          - bit_wise=True: Transmissor recebe bits, receptor sigmoid, BCE from_logits=False.
          - bit_wise=False: Transmissor one-hot, receptor softmax, CCE.
        """

        bits = self.rng.uniform([batch_size, self.k], 0, 2, tf.int32)

        # ------------------------------------------------------------------ #
        # Modo BMI: constelação treinável 2D + receptor bit-wise com logits  #
        # ------------------------------------------------------------------ #
        if self.bmi:
            indices = self.bits_to_indices(bits)
            one_hot = tf.one_hot(indices, depth=self.M)
            z = self.transmitter(one_hot)         # (batch, 2)
            y = self.add_GaussianNoise(z, ebno_db)
            logits = self.receiver(y)             # (batch, k) — LLRs

            if self.is_training:
                bits_float = tf.cast(bits, tf.float32)
                return self.bce(bits_float, logits)
            else:
                # Limiar em 0: logit > 0 ↔ sigmoid > 0.5
                bits_hat = tf.cast(tf.math.greater_equal(logits, 0.0), tf.int32)
                return bits, bits_hat

        # ------------------------------------------------------------------ #
        # Modos originais: bit-wise ou symbol-wise                           #
        # ------------------------------------------------------------------ #
        if self.bit_wise:
            z = self.transmitter(bits)
        else:
            indices = self.bits_to_indices(bits)
            one_hot = tf.one_hot(indices, depth=self.M)
            z = self.transmitter(one_hot)

        y = self.add_GaussianNoise(z, ebno_db)
        recev = self.receiver(y)

        if self.is_training:
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