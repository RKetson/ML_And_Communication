import libs.tf_config
import pickle
import tensorflow as tf
from IPython import display
from sionna.phy.utils import sim_ber


def train(model_train, snr_dB_Train, optimizer, epochs, batchs, local_weights,
          aval_training=True, steps_for_aval=1000, local_aval="./Buffer/aval_training"):
    """
    Função para treinamento do modelo.

    Obs.: O modelo é treinado dentro do escopo da função. Para acessar os pesos treinados,
    busque o arquivo no caminho indicado em `local_weights`.

    A seleção de dispositivo (GPU/CPU) é gerenciada pela estratégia de distribuição
    (MirroredStrategy ou padrão) configurada nos scripts principais. Não é necessário
    usar `tf.device` aqui, pois isso quebraria a compatibilidade com múltiplas GPUs.

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

    for i in range(epochs):
        # Forward pass — computa a loss dentro do GradientTape
        with tf.GradientTape() as tape:
            loss = model_train(batchs, snr_dB_Train)

        # Backward pass — computa e aplica os gradientes
        grads = tape.gradient(loss, model_train.trainable_weights)
        optimizer.apply_gradients(zip(grads, model_train.trainable_weights))

        # Progresso e snapshot de constelação
        if i % 100 == 0:
            display.clear_output(wait=True)
            print(f"{i}/{epochs}  Loss: {loss:.2E}")

            if i % steps_for_aval == 0 and aval_training:
                x = model_train.points_Constellation()
                data_const.append(x)
                with open(local_aval, 'wb') as f:
                    pickle.dump(data_const, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Salva os pesos finais em disco
    weights = model_train.get_weights()
    with open(local_weights, 'wb') as f:
        pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)


def recover_weights(model, local_weights):
    """
    Recupera os pesos treinados de um arquivo e retorna o modelo com os pesos carregados.

    Entradas:
        model:          Instância do modelo (sem pesos treinados).
        local_weights:  Caminho do arquivo de pesos gerado por train().

    Saída:
        model: Modelo com os pesos restaurados.
    """
    # Uma inferência dummy é necessária para construir as camadas antes de set_weights()
    model(tf.constant(1, tf.int32), tf.constant(10.0, tf.float32))

    with open(local_weights, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)

    return model


def aval_model(model, ebno_dbs, batch_size=127, block_errors=1000, max_iter=1000,
               graph_mode="xla", local=None):
    """
    Avalia um modelo já treinado via simulação de Monte Carlo (BER/SER).

    A seleção de dispositivo é gerenciada pela estratégia de distribuição configurada
    nos scripts principais. O `sim_ber` do Sionna respeita automaticamente o contexto
    de distribuição ativo — não é necessário (nem correto) usar `tf.device` aqui.

    Entradas:
        model:       Modelo a ser avaliado (modo inferência, is_training=False).
        ebno_dbs:    Array de valores de Eb/N0 (dB) a serem avaliados.
        batch_size:  Número de palavras-código processadas em paralelo por iteração MC.
        block_errors: Critério de parada: mínimo de blocos errados por ponto de SNR.
        max_iter:    Máximo de iterações Monte Carlo por ponto de SNR.
        graph_mode:  Modo de compilação: "xla", "graph" ou None.
        local:       Se fornecido, salva os dicionários BER/SER neste caminho.

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

    Entradas:
        local: Caminho do arquivo gerado por aval_model().

    Retorna:
        ber_dict: Dicionário {ebno_db: BER}.
        ser_dict: Dicionário {ebno_db: SER}.
    """
    with open(local, 'rb') as f:
        var = pickle.load(f)

    return var[0], var[1]