import os

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Respeita a variável de ambiente CUDA_HOME, com fallback para o caminho padrão
#_cuda_path = os.environ.get('CUDA_HOME', '/usr/lib/cuda')
#os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir={_cuda_path}'

import tensorflow as tf
# Avoid warnings from TensorFlow
#tf.get_logger().setLevel('ERROR')