import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.logging.set_verbosity(tf.logging.ERROR)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

