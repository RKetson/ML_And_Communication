import libs.tf_config
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

    NOTA DE GRAFO/XLA:
        O `call` desta camada é chamado dentro de funções anotadas com
        @tf.function(jit_compile=True) nos modelos. Todas as operações aqui
        (reduce_mean, reduce_sum, square, sqrt, divisão) são compatíveis com XLA.
    """
    def __init__(self, **kwargs):
        super(EnergyNormalization, self).__init__(**kwargs)

    def call(self, input):
        # Energia média por símbolo: média de ||x_i||^2 sobre o batch
        energy_avg = tf.reduce_mean(tf.reduce_sum(tf.square(input), axis=-1))
        # Normaliza para que E_s = 1 (energia média unitária)
        x_norm = input / tf.sqrt(energy_avg)
        return x_norm


class End2EndSystem(tf.keras.Model):
    """
    Modelo fim-a-fim com transmissor e receptor treináveis.

    Todos os caminhos de forward pass (treino e inferência) são compilados com
    @tf.function(jit_compile=True), habilitando XLA end-to-end. O loop de treino
    externo (GradientTape em val_model.train) invoca o __call__ já compilado, de
    modo que toda a computação de gradiente também beneficia da aceleração XLA.

    O método points_Constellation() é deliberadamente mantido fora do grafo XLA,
    pois é chamado apenas para visualização e não faz parte do caminho crítico.
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
        self.coderate = k / n

        self.rng = tf.random.get_global_generator()
        self.transmitter = tx
        self.receiver = rx
        self.bmi = bmi

        if bmi:
            self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        elif bit_wise:
            self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        else:
            self.bce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        self.is_training = training
        self.bit_wise = bit_wise

        # Tabela de potências de 2 pré-computada como constante do grafo.
        # Evita recriar o tensor a cada chamada dentro do XLA.
        self._powers = tf.constant(
            [2**i for i in range(k - 1, -1, -1)], dtype=tf.int32
        )

    # ---------------------------------------------------------------------- #
    # Operações auxiliares — todas compatíveis com XLA                        #
    # ---------------------------------------------------------------------- #

    def bits_to_indices(self, bits):
        """
        Converte matriz de bits (batch, k) em índices decimais (batch,).
        Compatível com XLA: usa apenas operações tensoriais puras.
        """
        # Inverte a ordem dos bits para MSB à esquerda
        binary_patches = tf.reverse(tf.reshape(bits, (-1, self.k)), axis=[-1])
        return tf.reduce_sum(
            binary_patches * (2 ** tf.range(self.k, dtype=tf.int32)), axis=-1
        )

    def indices_to_bits(self, indices):
        """
        Converte índices decimais (batch,) em matriz de bits (batch, k).
        Compatível com XLA: usa bitwise puras.
        """
        tensor_expanded = tf.cast(tf.expand_dims(indices, -1), tf.int32)
        powers = tf.range(self.k - 1, -1, -1)
        binary_tensor = tf.bitwise.bitwise_and(
            tf.bitwise.right_shift(tensor_expanded, powers), 1
        )
        return tf.reshape(binary_tensor, (-1, self.k))

    def convert_symbol_probs_to_bit_probs_graph_compatible(self, symbol_probs):
        """
        Converte probabilidades de símbolos (batch, M) em probabilidades de bits (batch, k).
        Usa a tabela de potências pré-computada (_powers) para evitar recriação no grafo.
        Compatível com XLA.
        """
        symbols_indices = tf.range(self.M, dtype=tf.int32)
        # bit_map[i, j] = 1 se o bit j do símbolo i é 1
        bit_flags = tf.bitwise.bitwise_and(
            tf.expand_dims(symbols_indices, axis=1),
            tf.expand_dims(self._powers, axis=0)
        )
        bit_map = tf.cast(tf.math.greater(bit_flags, 0), dtype=tf.float32)
        return tf.matmul(symbol_probs, bit_map)

    def add_GaussianNoise(self, y, ebno_db):
        """
        Adiciona ruído AWGN gaussiano ao sinal transmitido, parametrizado por Eb/N0.

        DERIVAÇÃO DA VARIÂNCIA DO RUÍDO:
            - Eb/N0 (linear) = ebno_linear
            - Taxa de código: R = k/n
            - Energia por símbolo: E_s = 1 (garantido pela EnergyNormalization)
            - Dimensões reais por símbolo: n
            - sigma^2 = N0/2 = 1 / (2 * ebno_linear * R * n)

        Compatível com XLA: todas as operações são elementares sobre tensores.
        """
        ebno_linear = tf.pow(10.0, ebno_db / 10.0)
        noise_variance = 1.0 / (2.0 * self.coderate * ebno_linear * self.n)
        noise = tf.random.normal(
            shape=tf.shape(y),
            mean=0.0,
            stddev=tf.sqrt(noise_variance),
            dtype=y.dtype
        )
        return tf.add(y, noise)

    # ---------------------------------------------------------------------- #
    # Forward pass compilado com XLA (jit_compile=True)                       #
    # ---------------------------------------------------------------------- #

    @tf.function(jit_compile=True)
    def __call__(self, batch_size, ebno_db):
        """
        Realiza 'batch_size' transmissões com SNR 'ebno_db'.

        Retorna:
          - Modo treino:     escalar de loss (float32)
          - Modo inferência: tupla (bits, bits_hat) de inteiros

        O decorator @tf.function(jit_compile=True) compila toda a função para XLA,
        incluindo transmissor, canal, receptor e cálculo de loss/erro em um único
        kernel de hardware — eliminando overhead de lançamento de kernels separados.

        NOTA SOBRE if/else Python vs tf.cond:
            Os branches if self.bmi / if self.bit_wise são resolvidos em tempo de
            traçagem do grafo (tracing-time), não em runtime. Portanto, são branches
            Python legítimos que o XLA dobra em especializações do grafo — sem
            custo de runtime, sem necessidade de tf.cond.
        """
        bits = self.rng.uniform([batch_size, self.k], 0, 2, tf.int32)

        # ------------------------------------------------------------------ #
        # Modo BMI: constelação treinável 2D + receptor bit-wise com logits   #
        # ------------------------------------------------------------------ #
        if self.bmi:
            indices = self.bits_to_indices(bits)
            one_hot = tf.one_hot(indices, depth=self.M)
            z       = self.transmitter(one_hot)
            y       = self.add_GaussianNoise(z, ebno_db)
            logits  = self.receiver(y)

            if self.is_training:
                bits_float = tf.cast(bits, tf.float32)
                return self.bce(bits_float, logits)
            else:
                # Limiar em 0: logit ≥ 0 ↔ sigmoid ≥ 0.5
                return bits, tf.cast(tf.math.greater_equal(logits, 0.0), tf.int32)

        # ------------------------------------------------------------------ #
        # Modos originais: bit-wise ou symbol-wise                            #
        # ------------------------------------------------------------------ #
        if self.bit_wise:
            z = self.transmitter(bits)
        else:
            indices = self.bits_to_indices(bits)
            one_hot = tf.one_hot(indices, depth=self.M)
            z       = self.transmitter(one_hot)

        y     = self.add_GaussianNoise(z, ebno_db)
        recev = self.receiver(y)

        if self.is_training:
            loss = self.bce(bits, recev) if self.bit_wise else self.bce(one_hot, recev)
            return loss
        else:
            if self.bit_wise:
                bits_hat = tf.cast(tf.math.greater_equal(recev, 0.5), tf.int32)
                return bits, bits_hat
            else:
                return bits, self.indices_to_bits(tf.math.argmax(recev, axis=-1))

    # ---------------------------------------------------------------------- #
    # Visualização — mantida fora do grafo XLA intencionalmente               #
    # ---------------------------------------------------------------------- #

    def points_Constellation(self, bits=None):
        """
        Retorna os pontos da constelação gerados pelo transmissor (sem ruído).

        NOTA: Este método é propositalmente NÃO decorado com @tf.function porque:
          1. É chamado apenas para visualização/monitoramento, não no caminho crítico.
          2. Retorna tensores que são convertidos para numpy imediatamente pelo caller.
          3. Operar em modo eager aqui é mais flexível e não tem impacto de performance.
        """
        if bits is None:
            numeros = tf.range(2**self.k, dtype=tf.int32)
            bits = tf.reverse(
                tf.bitwise.right_shift(numeros[:, tf.newaxis],
                                       tf.range(self.k, dtype=tf.int32)) & 1,
                axis=[1]
            )

        if self.bmi or not self.bit_wise:
            indices = self.bits_to_indices(bits)
            one_hot = tf.one_hot(indices, depth=self.M)
            z       = self.transmitter(one_hot)
        else:
            z = self.transmitter(bits)

        if bits is None or self.bmi or not self.bit_wise:
            return bits, z
        return z

    def calcular_distancias_ordem_n(self, z):
        """Calcula matriz de distâncias euclidianas entre pontos da constelação."""
        from scipy.spatial.distance import pdist, squareform
        if isinstance(z, tf.Tensor):
            z = z.numpy()
        return squareform(pdist(z, metric='euclidean'))

    def plot_transmitter(self, vizinhos=None):
        """Plota a constelação aprendida pelo transmissor."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 7))
        bits, z = self.points_Constellation()
        labels_binarios_str = [
            "".join(str(int(bit)) for bit in row) for row in bits.numpy()
        ]

        if vizinhos is not None:
            dist_eucled = self.calcular_distancias_ordem_n(z)

        def _plot(z_np, label=True):
            for i, point in enumerate(z_np):
                x, y = point
                ax.plot(x, y, 'bo')
                if label:
                    ax.annotate(labels_binarios_str[i], (x, y),
                                textcoords="offset points", xytext=(5, 5),
                                ha='center', fontsize=9)

        z_np = z.numpy()
        if z_np.shape[-1] == 2:
            _plot(z_np)
        else:
            from sklearn.manifold import TSNE
            from sklearn.preprocessing import MinMaxScaler
            tsne = TSNE(n_components=2, perplexity=2**self.k - 1, random_state=42)
            z_transform = MinMaxScaler().fit_transform(tsne.fit_transform(z_np))
            _plot(z_transform, label=False)

        if vizinhos is not None:
            z_plot = z_np if z_np.shape[-1] == 2 else z_transform
            for i in range(len(z_plot)):
                indices_viz = np.argsort(dist_eucled[i])[1:vizinhos + 1]
                print("#" * 40)
                for j in indices_viz:
                    print(f"Distância entre {labels_binarios_str[i]} e "
                          f"{labels_binarios_str[j]}: {dist_eucled[i, j]:.4f}")

        ax.set_title('Transmissor')
        ax.set_xlabel('Parte Real')
        ax.set_ylabel('Parte Imaginária')
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()