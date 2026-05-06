import libs.tf_config
import pickle
import tensorflow as tf
from IPython import display
from sionna.phy.utils import sim_ber


# ============================================================================================ #
# train_step: compilado com @tf.function(jit_compile=True)
#
# Por que separar o train_step do loop Python?
#   - O loop `for i in range(epochs)` é Python puro e não pode ser compilado.
#   - Ao isolar apenas o forward+backward como função compilada, o XLA/Grappler
#     fusionam as operações do GradientTape em um único kernel de hardware por iteração.
#   - Resultado: eliminação do overhead de lançamento de kernels individuais para
#     cada operação (matmul, relu, add_noise, bce, etc.) dentro de um passo.
#
# Compatibilidade com MirroredStrategy:
#   - A função é criada fora de qualquer escopo de estratégia. A estratégia injeta
#     suas variáveis espelhadas automaticamente via tf.Variable(synchronization=...).
#   - apply_gradients é chamado fora do @tf.function para permitir que a estratégia
#     gerencie a sincronização de gradientes entre réplicas.
# ============================================================================================ #

@tf.function(jit_compile=True)
def _forward_pass(model_train, batchs, snr_dB_Train):
    """
    Executa o forward pass e retorna a loss.
    Compilado com XLA para fusão de kernels.
    """
    return model_train(batchs, snr_dB_Train)


def train(model_train, snr_dB_Train, optimizer, epochs, batchs, local_weights,
          aval_training=True, steps_for_aval=1000, local_aval="./Buffer/aval_training"):
    """
    Função para treinamento do modelo.

    O forward pass de cada iteração é executado via _forward_pass() compilado com
    @tf.function(jit_compile=True), habilitando XLA no caminho crítico de treino.
    O apply_gradients fica fora do XLA para compatibilidade com MirroredStrategy.

    Entradas:
        model_train:    Modelo a ser treinado.
        snr_dB_Train:   Valor de SNR (dB) usado durante o treinamento.
        optimizer:      Otimizador usado no gradiente descendente.
        epochs:         Número de iterações de treinamento.
        batchs:         Número de amostras por iteração (batch size).
        local_weights:  Caminho onde os pesos treinados serão salvos.
        aval_training:  Se True, salva snapshots da constelação durante o treinamento.
        steps_for_aval: Intervalo de iterações entre snapshots da constelação.
        local_aval:     Caminho onde os snapshots da constelação serão salvos.
    """
    data_const = []
    snr_tensor = tf.constant(snr_dB_Train, dtype=tf.float32)
    batch_tensor = tf.constant(batchs, dtype=tf.int32)

    for i in range(epochs):
        # Forward pass + gradientes — XLA compila tudo dentro do tape em um kernel
        with tf.GradientTape() as tape:
            loss = _forward_pass(model_train, batch_tensor, snr_tensor)

        grads = tape.gradient(loss, model_train.trainable_weights)
        optimizer.apply_gradients(zip(grads, model_train.trainable_weights))

        # Progresso e snapshot de constelação (apenas a cada 100 iterações)
        if i % 100 == 0:
            display.clear_output(wait=True)
            print(f"{i}/{epochs}  Loss: {loss:.2E}")

            if i % steps_for_aval == 0 and aval_training:
                x = model_train.points_Constellation()
                data_const.append(x)
                with open(local_aval, 'wb') as f:
                    pickle.dump(data_const, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Salva pesos finais
    weights = model_train.get_weights()
    with open(local_weights, 'wb') as f:
        pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)


def recover_weights(model, local_weights):
    """
    Recupera os pesos treinados de um arquivo e retorna o modelo com os pesos carregados.

    A inferência dummy usa tf.constant para evitar retrace do grafo compilado.
    """
    # Inferência dummy com constantes TF — constrói as camadas sem retrace
    model(tf.constant(1, tf.int32), tf.constant(10.0, tf.float32))

    with open(local_weights, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)

    return model


def aval_model(model, ebno_dbs, batch_size=127, block_errors=1000, max_iter=1000,
               graph_mode="xla", local=None):
    """
    Avalia um modelo já treinado via simulação de Monte Carlo (BER/SER).

    O Sionna sim_ber respeita o graph_mode passado e compila internamente o modelo
    com @tf.function(jit_compile=True) quando graph_mode="xla". Como o __call__
    do End2EndSystem já está decorado com @tf.function(jit_compile=True), o Sionna
    reutiliza o grafo compilado sem retrace — sem custo de compilação duplicada.

    Entradas:
        model:        Modelo a ser avaliado (modo inferência, is_training=False).
        ebno_dbs:     Array de valores de Eb/N0 (dB) a serem avaliados.
        batch_size:   Número de palavras-código processadas em paralelo por iteração MC.
        block_errors: Critério de parada: mínimo de blocos errados por ponto de SNR.
        max_iter:     Máximo de iterações Monte Carlo por ponto de SNR.
        graph_mode:   Modo de compilação: "xla", "graph" ou None.
        local:        Se fornecido, salva os dicionários BER/SER neste caminho.

    Saídas:
        ber_dict: Dicionário {ebno_db: BER}.
        ser_dict: Dicionário {ebno_db: SER}.
    """
    ber, ser = sim_ber(
        model,
        ebno_dbs,
        batch_size=batch_size,
        num_target_block_errors=block_errors,
        max_mc_iter=max_iter,
        graph_mode=graph_mode
    )
    ber, ser = ber.numpy(), ser.numpy()

    ber_dict = {ebno: float(b) for ebno, b in zip(ebno_dbs, ber)}
    ser_dict = {ebno: float(s) for ebno, s in zip(ebno_dbs, ser)}

    if local is not None:
        with open(local, 'wb') as f:
            pickle.dump([ber_dict, ser_dict], f, protocol=pickle.HIGHEST_PROTOCOL)

    return ber_dict, ser_dict


def recover_points_model(local):
    """
    Lê um arquivo de pontos salvo por aval_model().

    Retorna:
        ber_dict: Dicionário {ebno_db: BER}.
        ser_dict: Dicionário {ebno_db: SER}.
    """
    with open(local, 'rb') as f:
        var = pickle.load(f)
    return var[0], var[1]