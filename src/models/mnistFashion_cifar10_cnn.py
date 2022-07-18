# %%
from imblearn import metrics
import tensorflow as tf
import pandas as pd
import numpy as np 
from functools import reduce
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
from imblearn.metrics import geometric_mean_score

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#only works for one-hot binary
class geometric_mean(tf.keras.metrics.Metric):
  def __init__(self, name='gm', **kwargs):
    super(geometric_mean, self).__init__(name=name, **kwargs)
    self.gmean = self.add_weight(name='gm', initializer='zeros')
    self.true_positives = self.add_weight(name='true_positive', initializer='zeros')
    self.true_negatives = self.add_weight(name='true_negatives', initializer='zeros')
    self.false_positives = self.add_weight(name='false_positive', initializer='zeros')
    self.false_negatives = self.add_weight(name='false_negatives', initializer='zeros')
  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(tf.gather(y_true, 0, axis=1), tf.bool)
    y_pred = tf.cast(tf.math.round(tf.gather(y_pred, 0, axis=1)), tf.bool)
    self.true_positives.assign_add(tf.math.reduce_sum(tf.cast(tf.equal(tf.gather(y_pred,tf.where(tf.equal(y_true, True))),True),tf.float32)))
    self.false_positives.assign_add(tf.math.reduce_sum(tf.cast(tf.equal(tf.gather(y_pred,tf.where(tf.equal(y_true, True))),False),tf.float32)))
    self.true_negatives.assign_add(tf.math.reduce_sum(tf.cast(tf.equal(tf.gather(y_pred,tf.where(tf.equal(y_true, False))),False),tf.float32)))
    self.false_negatives.assign_add(tf.math.reduce_sum(tf.cast(tf.equal(tf.gather(y_pred,tf.where(tf.equal(y_true, False))),True),tf.float32)))
  def result(self):
    #   Sensitivity = TruePositive / (TruePositive + FalseNegative)
    sens = tf.math.divide_no_nan(self.true_positives, (self.true_positives+self.false_negatives))
    # Specificity = TrueNegative / (FalsePositive + TrueNegative)
    spec = tf.math.divide_no_nan(self.true_negatives, (self.false_positives+self.true_negatives))
    # G-Mean = sqrt(Sensitivity * Specificity)
    gm = tf.math.multiply(sens,spec)
    self.gmean.assign(tf.sqrt(gm))
    return self.gmean
  def reset_states(self): 
    # The state of the metric will be reset at the start of each epoch.
    self.gmean.assign(0.0)
    self.true_positives.assign(0.0)
    self.true_negatives.assign(0.0)
    self.false_positives.assign(0.0)
    self.false_negatives.assign(0.0)


METRICS = [
      geometric_mean(),
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

#The models
def get_modelCnnSmall(inputDim, outputDim, depth=1, hidden=10, useDp=False, modelName='best_model'):
    inp = tf.keras.Input(inputDim)
    if useDp:
        x = tf.keras.layers.Dropout(rate=0.2,noise_shape=inputDim)(inp)
        x = tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),padding="same", activation='relu')(x)
    else:
        x = tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),padding="same", activation='relu')(inp)
    x = tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),padding="same", kernel_initializer='he_uniform', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    if depth > 1:
        x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding="same", activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding="same", activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    if depth > 2:
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    if depth > 3:
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    if depth > 4:
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    if useDp:
        x = tf.keras.layers.Dropout(rate=0.2,noise_shape=x.shape)(x)
    x = tf.keras.layers.Dense(hidden, activation='relu')(x)
    x = tf.keras.layers.Dense(hidden, activation='relu')(x)
    out = tf.keras.layers.Dense(outputDim, activation='softmax')(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(loss='binary_crossentropy', metrics=METRICS, optimizer=tf.keras.optimizers.Adam())
    return model
