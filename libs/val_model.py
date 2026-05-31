import os
import pickle
import numpy as np
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
          aval_training=True, steps_for_aval=1000, local_aval="./Buffer/aval_training",
          generate_gif=False, gif_path="./Figures/training_evolution.gif",
          gif_title="Evolução da Constelação", steps_per_epoch=1):
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
        epoch_loss = 0.0
        # _train_step: forward + gradientes num único grafo compilado (@tf.function)
        # apply_gradients fora para compatibilidade com MirroredStrategy (AllReduce)
        for _ in range(steps_per_epoch):
            loss, grads = _train_step(model_train, batch_tensor, snr_tensor)
            optimizer.apply_gradients(zip(grads, model_train.trainable_weights))
            epoch_loss += loss.numpy()
            
        current_loss = epoch_loss / steps_per_epoch

        # Progresso e snapshot de constelação (apenas a cada 100 iterações)
        if i % 100 == 0:
            display.clear_output(wait=True)
            print(f"{i}/{epochs}  Loss: {current_loss:.2E}")

            if i % steps_for_aval == 0 and aval_training:
                x = model_train.points_Constellation()
                data_const.append({'snr': snr_dB_Train, 'points': x})
                os.makedirs(os.path.dirname(local_aval), exist_ok=True)
                with open(local_aval, 'wb') as f:
                    pickle.dump(data_const, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Salva pesos finais
    weights = model_train.get_weights()
    with open(local_weights, 'wb') as f:
        pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)

    if aval_training and generate_gif:
        generate_training_gif(local_aval, gif_path, title=gif_title)


def train_curriculum(model_train, snr_start, snr_end, snr_step, patience, optimizer, epochs, batchs, local_weights,
                     aval_training=True, steps_for_aval=1000, local_aval="./Buffer/aval_training",
                     generate_gif=False, gif_path="./Figures/training_evolution.gif",
                     gif_title="Evolução da Constelação", steps_per_epoch=1):
    """
    Função de treinamento com Curriculum Learning (SNR dinâmico baseado em estagnação).
    """
    data_const = []
    current_snr = snr_start
    snr_tensor = tf.constant(current_snr, dtype=tf.float32)
    batch_tensor = tf.constant(batchs, dtype=tf.int32)
    
    best_loss = float('inf')
    wait = 0
    ema_loss = None
    
    is_decreasing = snr_start > snr_end
    
    for i in range(epochs):
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            loss, grads = _train_step(model_train, batch_tensor, snr_tensor)
            optimizer.apply_gradients(zip(grads, model_train.trainable_weights))
            epoch_loss += loss.numpy()
            
        current_loss = epoch_loss / steps_per_epoch
        
        # Atualiza a Média Móvel Exponencial (EMA)
        if ema_loss is None:
            ema_loss = current_loss
        else:
            ema_loss = 0.99 * ema_loss + 0.01 * current_loss
            
        # Verifica estagnação da loss (melhoria de pelo menos 0.1% relativa à grandeza atual)
        min_improvement = ema_loss * 0.0001
        if ema_loss < best_loss - min_improvement:
            best_loss = ema_loss
            wait = 0
        else:
            wait += 1
            
        # Se esgotou a paciência
        if wait >= patience:
            if is_decreasing and current_snr > snr_end:
                current_snr -= snr_step
                current_snr = max(current_snr, snr_end)
                snr_tensor = tf.constant(current_snr, dtype=tf.float32)
                wait = 0
                best_loss = float('inf')  # Reseta o melhor para se readaptar ao novo ruído
                ema_loss = None
            elif not is_decreasing and current_snr < snr_end:
                current_snr += snr_step
                current_snr = min(current_snr, snr_end)
                snr_tensor = tf.constant(current_snr, dtype=tf.float32)
                wait = 0
                best_loss = float('inf')  # Reseta o melhor para se readaptar ao novo ruído
                ema_loss = None
            else:
                print(f"\nTreinamento concluído antecipadamente (Early Stopping) no SNR final de {current_snr:.1f} dB na iteração {i}.")
                break
            
        if i % 100 == 0:
            display.clear_output(wait=True)
            print(f"{i}/{epochs}  SNR: {current_snr:.1f} dB  Loss: {current_loss:.2E} (EMA: {ema_loss:.2E})  Wait: {wait}/{patience}")

        if i % steps_for_aval == 0 and aval_training:
            x = model_train.points_Constellation()
            data_const.append({'snr': current_snr, 'points': x})
            os.makedirs(os.path.dirname(local_aval), exist_ok=True)
            with open(local_aval, 'wb') as f:
                pickle.dump(data_const, f, protocol=pickle.HIGHEST_PROTOCOL)

    weights = model_train.get_weights()
    with open(local_weights, 'wb') as f:
        pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)

    if aval_training and generate_gif:
        generate_training_gif(local_aval, gif_path, title=gif_title)


def generate_training_gif(local_aval, gif_path, title="Evolução da Constelação"):
    """
    Gera um GIF animado a partir do histórico de constelações salvo durante o treinamento.
    Se a dimensionalidade for maior que 2, utiliza PCA para reduzir para 2D.
    """
    import io
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from sklearn.manifold import TSNE

    if not os.path.exists(local_aval):
        print(f"Arquivo de histórico {local_aval} não encontrado. O GIF não será gerado.")
        return

    with open(local_aval, 'rb') as f:
        data_const = pickle.load(f)

    if len(data_const) == 0:
        print("Histórico vazio. O GIF não será gerado.")
        return

    # Garante que o diretório de destino existe
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    frames = []
    
    # Extrai o último frame para entender a dimensionalidade
    last_frame = data_const[-1]
    
    if isinstance(last_frame, dict):
        last_frame_points = last_frame.get('points')
    else:
        last_frame_points = last_frame

    if isinstance(last_frame_points, tuple) and len(last_frame_points) == 2:
        _, last_z = last_frame_points
    else:
        last_z = last_frame_points

    if isinstance(last_z, tf.Tensor):
        last_z_np = last_z.numpy()
    else:
        last_z_np = last_z

    n_dims = last_z_np.shape[-1]
    
    def normalize_tsne(coords):
        energy_avg = np.mean(np.sum(np.square(coords), axis=-1))
        return coords / np.sqrt(energy_avg)

    if n_dims > 2:
        perplexity = min(5, len(last_z_np) - 1)
        tsne_model = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca')
        last_z_plot = tsne_model.fit_transform(last_z_np)
        last_z_plot = normalize_tsne(last_z_plot)
    else:
        last_z_plot = last_z_np
        
    lim_max = np.max(np.abs(last_z_plot)) + 0.5
        
    for idx, frame_data in enumerate(data_const):
        if isinstance(frame_data, dict):
            snr = frame_data.get('snr')
            points_data = frame_data.get('points')
        else:
            snr = None
            points_data = frame_data

        if isinstance(points_data, tuple) and len(points_data) == 2:
            bits, z = points_data
        else:
            bits, z = None, points_data
            
        if isinstance(z, tf.Tensor):
            z_np = z.numpy()
        else:
            z_np = z
            
        if bits is not None:
            if isinstance(bits, tf.Tensor):
                bits_np = bits.numpy()
            else:
                bits_np = bits
            labels_binarios_str = ["".join(str(int(bit)) for bit in row) for row in bits_np]
        else:
            k_val = int(np.log2(len(z_np)))
            labels_binarios_str = [f"{i:0{k_val}b}" for i in range(len(z_np))]
            
        if n_dims > 2:
            tsne_model = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca')
            z_plot = tsne_model.fit_transform(z_np)
            z_plot = normalize_tsne(z_plot)
        else:
            z_plot = z_np

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(z_plot[:, 0], z_plot[:, 1], c='b')
        
        for i, point in enumerate(z_plot):
            ax.annotate(labels_binarios_str[i], (point[0], point[1]),
                        textcoords="offset points", xytext=(5, 5),
                        ha='center', fontsize=9)
            
        title_str = f'{title} - Passo {idx+1}'
        if snr is not None:
            title_str += f' (SNR: {snr:.1f} dB)'
            
        ax.set_title(title_str)
        ax.grid(True)
        ax.set_xlim(-lim_max, lim_max)
        ax.set_ylim(-lim_max, lim_max)
        ax.set_aspect('equal', adjustable='box')
        
        # Salva em memória
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf))

    if frames:
        frames[0].save(
            gif_path,
            format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=100, 
            loop=0
        )
        print(f"GIF da evolução salvo em: {gif_path}")


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