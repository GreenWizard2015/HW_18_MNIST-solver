import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import glob
import os
import KTUtils

def createOptimizer(name, lr):
  if 'adam' == name:
    return optimizers.Adam(lr=lr)
  if 'sgd' == name:
    return optimizers.SGD(lr=lr)
  if 'rmsprop' == name:
    return optimizers.RMSprop(lr=lr)
  return

def commonDenseModel(LReLuSlope=0.0, useBN=False, dropout=0, hidden_layers=[], optimizer=None):
  inputX = res = layers.Input(shape=(28, 28))
  res = layers.Flatten()(res)
  
  for sz in hidden_layers:
    if useBN:
      res = layers.BatchNormalization()(res)
    res = layers.Dense(sz, activation=None)(res)
    res = layers.LeakyReLU(alpha=LReLuSlope)(res)
    if 0 < dropout:
     res = layers.Dropout(.1)(res)
  
  res = layers.Dense(10, activation='softmax')(res)

  model = keras.Model(inputs=[inputX], outputs=[res])
  model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
  )
  return model

def V1(hp):
  name = 'tunable_V1'
  assert name == hp.Choice('ModelName', [name])
  
  MAX_DEPTH = 5
  hidden_layers = []
  for i in range(hp.Int('depth', 2, MAX_DEPTH, default=MAX_DEPTH)):
    sz = hp.Int(
      'layer_%d' % i, 16, 512, step=32,
      parent_name='depth', parent_values=list(range(i + 1, MAX_DEPTH + 1))
    )
    hidden_layers.append(sz)
  
  return commonDenseModel(
    LReLuSlope=hp.Float('LReLuSlope', -1.0, 0.0, 0.1),
    hidden_layers=hidden_layers,
    optimizer=createOptimizer('adam', 1e-3),
    useBN=hp.Boolean('useBN'),
    dropout=hp.Float('dropout', 0.0, 0.5, 0.1)
  )

def allModels(hp):
  modelsBuilders = {
    'tunable_V1': V1,
  }
  ModelName = hp.Choice('ModelName', list(modelsBuilders.keys()))
  return modelsBuilders[ModelName](hp)

def fromFolder(folder):
  res = {}
  for modelJson in glob.iglob(os.path.join(folder, '*.json')):
    name = os.path.basename(modelJson).replace('.json', '')
    HP = KTUtils.loadTrial(modelJson)
    res[name] = lambda: allModels(HP)

  return res