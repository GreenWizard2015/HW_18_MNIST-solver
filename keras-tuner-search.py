#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Utils
Utils.setup(MAX_GPU_MEMORY=2 * 1024, RANDOM_SEED=1671)

import numpy as np
import os

import tensorflow as tf

import MnistDataset
import TunableModels
import kerastuner.tuners as kt
import KTUtils
from kerastuner.engine.oracle import Objective

TOP_MODELS_N = 10

SEARCH_ITERATIONS = 3
NB_EPOCH = 30

BATCH_SIZE = 128
VERBOSE = 2
VALIDATION_SPLIT = 0.2 # how much TRAIN is reserved for VALIDATION
ES_PATIENCE = 3 # early stopping patience
RESTORE_BEST = False # just stop training
MONITOR_VAL = 'val_loss'
MONITOR_MODE = 'min'

FOLDER = os.path.join(os.path.dirname(__file__), 'tuner')
filepath = lambda *x: os.path.join(FOLDER, *x)

(X_train, Y_train), (X_test, Y_test), (X_val, Y_val) = MnistDataset.dataset(VALIDATION_SPLIT)
#######################
tuner = kt.Hyperband(
  TunableModels.V1,
  objective=Objective(MONITOR_VAL, MONITOR_MODE),
  max_epochs=NB_EPOCH,
  factor=2,
  hyperband_iterations=SEARCH_ITERATIONS,
  executions_per_trial=1,
  directory=filepath('tuner-data'),
  overwrite=True,
)

tuner.search_space_summary()
tuner.search(
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
  ]
)

tuner.results_summary(TOP_MODELS_N)
for i, candidate in enumerate(tuner.get_best_hyperparameters(TOP_MODELS_N)):
  KTUtils.saveTrial(candidate, saveTo=lambda uid: filepath('%02d.json' % (i)))