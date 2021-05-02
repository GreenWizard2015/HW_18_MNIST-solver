from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop

def baselineModels():
  return {
    'V1': baselineV1,
    'V2': baselineV2,
    'V3': baselineV3,
    'V4': baselineV4,
  }

def baselineV1(shape=(28, 28), NB_CLASSES=10):
  model = Sequential()
  model.add(Flatten(input_shape=shape)) # convert to raw input
  
  model.add(Dense(NB_CLASSES))
  model.add(Activation('softmax'))
  
  model.compile(
    optimizer=SGD(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
  )
  return model

def baselineV2(shape=(28, 28), NB_CLASSES=10, N_HIDDEN=128):
  model = Sequential()
  model.add(Flatten(input_shape=shape)) # convert to raw input
  
  model.add(Dense(N_HIDDEN))
  model.add(Activation('relu'))
  model.add(Dense(N_HIDDEN))
  model.add(Activation('relu'))
  model.add(Dense(NB_CLASSES))
  model.add(Activation('softmax'))

  model.compile(
    optimizer=SGD(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
  )
  return model

def baselineV3(shape=(28, 28), NB_CLASSES=10, N_HIDDEN=128, DROPOUT=0.3):
  model = Sequential()
  model.add(Flatten(input_shape=shape)) # convert to raw input

  model.add(Dense(N_HIDDEN))
  model.add(Activation('relu'))
  model.add(Dropout(DROPOUT))
  model.add(Dense(N_HIDDEN))
  model.add(Activation('relu'))
  model.add(Dropout(DROPOUT))  # прореживание
  model.add(Dense(NB_CLASSES))
  model.add(Activation('softmax'))
  
  model.compile(
    optimizer=SGD(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
  )
  return model

def baselineV4(shape=(28, 28), NB_CLASSES=10, N_HIDDEN=128, DROPOUT=0.3):
  model = Sequential()
  model.add(Flatten(input_shape=shape)) # convert to raw input

  model.add(Dense(N_HIDDEN))
  model.add(Activation('relu'))
  model.add(Dropout(DROPOUT))
  model.add(Dense(N_HIDDEN))
  model.add(Activation('relu'))
  model.add(Dropout(DROPOUT))
  model.add(Dense(NB_CLASSES))
  model.add(Activation('softmax'))
  
  model.compile(
    optimizer=RMSprop(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
  )
  return model