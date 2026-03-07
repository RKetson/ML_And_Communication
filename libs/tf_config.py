import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'

import tensorflow as tf
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')