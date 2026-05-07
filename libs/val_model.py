import libs.tf_config
import pickle
import tensorflow as tf
from IPython import display
from sionna.phy.utils import sim_ber


# ============================================================================================ #
# _train_step: compilado com @tf.function
#
# POR QUE O GradientTape DEVE ESTAR DENTRO DA MESMA @tf.function DO FORWARD PASS?
#
#   Quando separamos o forward pass em um @tf.function(jit_compile=True) independente
#   e chamamos de dentro de um GradientTape externo, o XLA compila o forward como um
#   bloco opaco — a tape não consegue rastrear operações DENTRO do bloco XLA compilado,
#   retornando gradientes None para todas as variáveis.
#
#   A solução correta (padrão canônico TF2) é colocar tape + forward + gradient dentro
#   de uma única @tf.function. O TF então traça o grafo completo forward+backward como
#   um único grafo diferenciável.
#
#   NOTA: Usamos @tf.function SEM jit_compile=True no _train_step externo porque:
#     1. GradientTape com jit_compile pode ter problemas com ops de grad customizadas.
#     2. O __call__ do modelo já tem @tf.function(jit_compile=True) — o XLA é aplicado
#        na parte computacionalmente intensiva (rede neural + canal), que é o que importa.
#     3. O @tf.function externo elimina o overhead Python entre iterações via graph mode.
#
#   Compatibilidade com MirroredStrategy:
#     apply_gradients fica FORA do @tf.function, permitindo que a estratégia gerencie
#     a sincronização de gradientes entre réplicas (AllReduce).
# ============================================================================================ #

@tf.function
def _train_step(model_train, batch_tensor, snr_tensor):
    """
    Executa um passo de treinamento: forward pass + cálculo de gradientes.
    Retorna (loss, grads) para que apply_gradients seja chamado fora,
    mantendo compatibilidade com MirroredStrategy.
    """
    with tf.GradientTape() as tape:
        loss = model_train(batch_tensor, snr_tensor)
    grads = tape.gradient(loss, model_train.trainable_weights)
    return loss, grads



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
        # _train_step: forward + gradientes num único grafo compilado (@tf.function)
        # apply_gradients fora para compatibilidade com MirroredStrategy (AllReduce)
        loss, grads = _train_step(model_train, batch_tensor, snr_tensor)
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