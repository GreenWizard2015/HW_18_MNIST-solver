from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dataset(VALIDATION_SPLIT):
  (X_train, y_train), (X_test, y_test) = mnist.load_data()
  # normalize
  X_train = X_train.astype('float32') / 255.0
  X_test = X_test.astype('float32') / 255.0
  # convert class vectors to binary class matrices
  NB_CLASSES = 10
  Y_train = to_categorical(y_train, NB_CLASSES)
  Y_test = to_categorical(y_test, NB_CLASSES)
  
  # split train into validation and train
  splitIndex = int(len(X_train) * VALIDATION_SPLIT)
  X_val = X_train[:splitIndex]
  Y_val = Y_train[:splitIndex]
  
  X_train = X_train[splitIndex:]
  Y_train = Y_train[splitIndex:]

  return (X_train, Y_train), (X_test, Y_test), (X_val, Y_val)