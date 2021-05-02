#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Utils
Utils.setup(MAX_GPU_MEMORY=2 * 1024, RANDOM_SEED=1671)

import baseline_models
import MnistDataset
import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import confusion_matrix

NB_EPOCH = 100
BATCH_SIZE = 128
VERBOSE = 2
VALIDATION_SPLIT = 0.2 # how much TRAIN is reserved for VALIDATION
ES_PATIENCE = 10 # early stopping patience
RESTORE_BEST = False # just stop training
MONITOR_VAL = 'val_loss'
MONITOR_MODE = 'min'

FOLDER = os.path.join(os.path.dirname(__file__), 'models')
filepath = lambda *x: os.path.join(FOLDER, *x)

MODELS = {}
MODELS.update(baseline_models.baselineModels()) # include baselines

modelsAccuracy = dict()
(X_train, Y_train), (X_test, Y_test) = MnistDataset.dataset()
# split train into validation and train
splitIndex = int(len(X_train) * VALIDATION_SPLIT)
X_val = X_train[:splitIndex]
Y_val = Y_train[:splitIndex]

X_train = X_train[splitIndex:]
Y_train = Y_train[splitIndex:]
#####
for modelName, modelBuilder in MODELS.items():
  print('Start training model "%s"' % modelName)
  Utils.setupRandomSeed(1671)
  
  model = modelBuilder()
  bestModelFileName = filepath(modelName, 'best.h5')
  os.makedirs(os.path.dirname(bestModelFileName), exist_ok=True)
  
  history = model.fit(
    X_train, Y_train,
    batch_size=BATCH_SIZE,
    epochs=NB_EPOCH,
    verbose=VERBOSE,
    validation_data=(X_val, Y_val),
    callbacks=[
      tf.keras.callbacks.EarlyStopping(
        monitor=MONITOR_VAL, mode=MONITOR_MODE, patience=ES_PATIENCE,
        restore_best_weights=RESTORE_BEST
      ),
      tf.keras.callbacks.ModelCheckpoint(
        filepath=bestModelFileName,
        save_best_only=True, save_weights_only=True,
        monitor=MONITOR_VAL, mode=MONITOR_MODE
      )
    ]
  ).history
  ######
  Utils.saveMetrics(
    history,
    lambda name: filepath(modelName, name),
    startEpoch=1
  )
  ######
  # evaluate best model
  model.load_weights(bestModelFileName)
  
  pred_labels = model.predict(X_test).argmax(axis=-1)
  CM = confusion_matrix(Y_test.argmax(axis=-1), pred_labels)
  accuracy = np.trace(CM) / float(np.sum(CM))
  modelsAccuracy[modelName] = accuracy

  Utils.plot_confusion_matrix(
    CM,
    target_names=[str(i) for i in range(10)],
    saveTo=filepath(modelName, 'confusion_matrix.png'),
    onlyErrors=True
  )
  print('Test accuracy: ', accuracy)
  print()
  continue

############
modelsByAcc = list(sorted(list(modelsAccuracy.items()), key=lambda x: -x[1]))
Utils.barPlot(
  [x[1] for x in modelsByAcc],
  [x[0] for x in modelsByAcc],
  'Accuracy',
  saveTo=filepath('accuracy.png')
)

print('Done')