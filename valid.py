import tensorflow as tf
from tensorflow import keras
from model.model import CRNN



validation_model = CRNN([], True)

validation_model.load_weights('best_model.hdf5')
