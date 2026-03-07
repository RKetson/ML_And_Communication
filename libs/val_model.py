import libs.tf_config
import pickle
import tensorflow as tf
from IPython import display
from sionna.phy.utils import sim_ber

from IPython import display

def train(model_train, snr_dB_Train, optimizer, epochs, batchs, local_weights, aval_training=True, steps_for_aval=1000, local_aval="./Buffer/aval_training"):
    """
        Função para treinamento do modelo.

        Obs.: Modelo é treinado dentro do escopo da função, para acessar aos pesos treinados,
        favor buscar os pesos treinados no arquivo de destino.

        Entradas:
            model_train: Modelo a ser treinado.
            snr_dB_Train: Valor de SNR usado durante o treinamento.
            optimizer: Otimizador usado no gradiente descendente.
            epochs: Número de épocas de treinamento.
            batchs: Número de batchs de treinamento por época.
            local_weights: Local onde serão armazenados os pesos treinados do modelo.
            aval_training: Deseja salvar o progresso do transmissor durante o treinamento?
            steps_for_aval: Épocas até salvar o progresso do transmissor.
            local_aval: Onde será armazenado os dados de transmissão durante o treinamento.

    """
    
    # Training loop
    with tf.device('/device:GPU:0'):
        data_const = []
        for i in range(epochs):
            # Forward pass
            with tf.GradientTape() as tape:
                loss = model_train(batchs, snr_dB_Train)

            # Computing and applying gradients
            grads = tape.gradient(loss, model_train.trainable_weights)
            optimizer.apply_gradients(zip(grads, model_train.trainable_weights))
            # Print progress
            
            if i % 100 == 0:
                display.clear_output(wait=True)
                print(f"{i}/{epochs}  Loss: {loss:.2E}")
                if i % steps_for_aval == 0 and aval_training == True:
                    x = model_train.points_Constellation()
                    data_const.append(x)
                    with open(local_aval, 'wb') as f:
                        pickle.dump(data_const, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save the weights in a file
    weights = model_train.get_weights()
    with open(local_weights, 'wb') as f:
        pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)


def recover_weights(model, local_weights):
    """
        Recupera os pesos treinados de um arquivo e retorna modelo treinado.
    """
    
    # Run one inference to build the layers and loading the weights
    model(tf.constant(1, tf.int32), tf.constant(10.0, tf.float32))

    with open(local_weights, 'rb') as f:
        weights = pickle.load(f)
        model.set_weights(weights)
    
    return model

def aval_model(model, ebno_dbs, batch_size=127, block_errors=1000, max_iter=1000, graph_mode="xla", local=None):
    """
        Avalia um modelo já treinado.

        Entradas:
            model: Modelo a ser avaliado.
            ebno_dbs: Lista de SNRs a qual o modelo será testado.
            batch_size: Processamentos em paralelo.
            block_errors: Critério de parada por SNR baseado no número mínimo de blocos de erro.
            max_iter: Máximo de iterações por SNR caso nenhum outro critério de parada tenha sido acionado.
            graph_mode: Modo de paralelização.

        Saídas:
            ber: Taxa de erro por bit a cada SNR avaliado.
            ser: Taxa de erro por símbolo a cada SNR avaliado.
    """
    with tf.device('/device:GPU:0'):
        ber, ser = sim_ber(model, ebno_dbs, batch_size=batch_size, num_target_block_errors=block_errors, max_mc_iter=max_iter, graph_mode=graph_mode)
        return ber.numpy(), ser.numpy()
    
    ber_dict = {ebno: b for ebno, b in zip(ebno_dbs, ber)}
    ser_dict = {ebno: s for ebno, s in zip(ebno_dbs, ser)}
    
    if local is not None:
        with open(local, 'wb') as f:
            pickle.dump([ber_dict, ser_dict], f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return ber_dict, ser_dict

def recover_points_model(local):
    with open(local, 'rb') as f:
        var = pickle.load(f)
        
    return var