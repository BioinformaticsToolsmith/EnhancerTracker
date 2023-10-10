from tensorflow import keras
import numpy as np
from keras import layers
import tensorflow as tf
import random
import loaders
from tensorflow.keras.metrics import Precision, Recall
import sys

K = keras.backend

################################################################################
# Models and layers
################################################################################


def make_conv_classifier(codings_size, shape, filter_1d_size=11, filter_num = 4):
    '''
    Function classifier that will be the baseline that does not utilize weight sharing for calculating distances.
    '''    
    kernel_2d_col = 3 
    
    inputs = keras.layers.Input(shape=shape)
    z = keras.layers.Masking(mask_value=0)(inputs)
    z = keras.layers.SeparableConv2D(filters=filter_num, kernel_size=(4,kernel_2d_col), use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)

    z = keras.layers.Reshape((shape[1]-(kernel_2d_col - 1), filter_num))(z)
    z = keras.layers.MaxPooling1D(pool_size=2)(z)

    z = keras.layers.SeparableConv1D(filters=filter_num*2, kernel_size=filter_1d_size, use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)

    z = keras.layers.MaxPooling1D(pool_size=2)(z)
    
    z = keras.layers.SeparableConv1D(filters=filter_num*4, kernel_size=filter_1d_size, use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)

    z = keras.layers.MaxPooling1D(pool_size=2)(z)
    
    z = keras.layers.SeparableConv1D(filters=filter_num*8, kernel_size=filter_1d_size, use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)

    z = keras.layers.MaxPooling1D(pool_size=2)(z)
    
    z = keras.layers.SeparableConv1D(filters=filter_num*16, kernel_size=filter_1d_size, use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)

    z = keras.layers.GlobalMaxPooling1D()(z)
    
    #z = keras.layers.Flatten()(z)
    
    z = keras.layers.Dense(codings_size, use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)
    
    outputs = keras.layers.Dense(1, activation='sigmoid')(z)
    return keras.Model(inputs = [inputs], outputs=[outputs])

def make_conv_classifier_no_batch(codings_size, shape, filter_1d_size=11, filter_num = 4):
    '''
    Function classifier that will be the baseline that does not utilize weight sharing for calculating distances.
    '''    
    kernel_2d_col = 3 
    
    inputs = keras.layers.Input(shape=shape)
    z = keras.layers.Masking(mask_value=0)(inputs)
    #z = keras.layers.SeparableConv2D(filters=filter_num, kernel_size=(4,kernel_2d_col), use_bias=True, activation='selu')(z)
    z = keras.layers.SeparableConv2D(filters=filter_num, kernel_size=(4,kernel_2d_col), use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)

    z = keras.layers.Reshape((shape[1]-(kernel_2d_col - 1), filter_num))(z)
    # z = keras.layers.MaxPooling1D(pool_size=2)(z)
    
    z = keras.layers.SeparableConv1D(filters=filter_num*2, kernel_size=filter_1d_size, use_bias=True, activation='selu')(z)
    z = keras.layers.SeparableConv1D(filters=filter_num*2, kernel_size=filter_1d_size, use_bias=False, strides=2)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)

    #z = keras.layers.MaxPooling1D(pool_size=2)(z)
    
    z = keras.layers.SeparableConv1D(filters=filter_num*4, kernel_size=filter_1d_size, use_bias=True, activation='selu')(z)
    z = keras.layers.SeparableConv1D(filters=filter_num*4, kernel_size=filter_1d_size, use_bias=False, strides=2)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)

    # z = keras.layers.MaxPooling1D(pool_size=2)(z)
    z = keras.layers.SeparableConv1D(filters=filter_num*8, kernel_size=filter_1d_size, use_bias=True, activation='selu')(z)
    z = keras.layers.SeparableConv1D(filters=filter_num*8, kernel_size=filter_1d_size, use_bias=False, strides=2)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)

    # z = keras.layers.MaxPooling1D(pool_size=2)(z)
    # z = keras.layers.SeparableConv1D(filters=filter_num*16, kernel_size=filter_1d_size, use_bias=True, activation='selu')(z)
    z = keras.layers.SeparableConv1D(filters=filter_num*16, kernel_size=filter_1d_size, use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)

    z = keras.layers.GlobalMaxPooling1D()(z)
    
    #z = keras.layers.Flatten()(z)
    
    z = keras.layers.Dense(codings_size, use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)
    
    outputs = keras.layers.Dense(1, activation='sigmoid')(z)
    return keras.Model(inputs = [inputs], outputs=[outputs])

def make_conv_classifier_monte(codings_size, shape, filter_1d_size=11, filter_num = 4):
    '''
    Function classifier that will be the baseline that does not utilize weight sharing for calculating distances.
    '''    

    class MCDropout(keras.layers.AlphaDropout):
        def call(self, inputs):
            return super().call(inputs, training=True)
    
    kernel_2d_col = 3 
    
    inputs = keras.layers.Input(shape=shape)
    z = keras.layers.Masking(mask_value=0)(inputs)
    z = keras.layers.SeparableConv2D(filters=filter_num, kernel_size=(4,kernel_2d_col), use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)

    z = keras.layers.Reshape((shape[1]-(kernel_2d_col - 1), filter_num))(z)
    z = keras.layers.MaxPooling1D(pool_size=2)(z)

    z = keras.layers.SeparableConv1D(filters=filter_num*2, kernel_size=filter_1d_size, use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)
    z = keras.layers.MaxPooling1D(pool_size=2)(z)
    
    z = keras.layers.SeparableConv1D(filters=filter_num*4, kernel_size=filter_1d_size, use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)
    z = keras.layers.MaxPooling1D(pool_size=2)(z)
    
    z = keras.layers.SeparableConv1D(filters=filter_num*8, kernel_size=filter_1d_size, use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)
    z = keras.layers.MaxPooling1D(pool_size=2)(z)
    
    z = keras.layers.SeparableConv1D(filters=filter_num*16, kernel_size=filter_1d_size, use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)
    z = keras.layers.GlobalMaxPooling1D()(z)
    
    # z = keras.layers.Dense(codings_size, use_bias=False)(z)
    # z = keras.layers.BatchNormalization()(z)
    # z = keras.layers.Activation(activation='selu')(z)
    
    z = keras.layers.Dense(codings_size, use_bias = False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)
    z = MCDropout(0.5)(z)

    z = keras.layers.Dense(codings_size, use_bias = False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)
    z = MCDropout(0.5)(z)

    outputs = keras.layers.Dense(1, activation='sigmoid')(z)
    return keras.Model(inputs = [inputs], outputs=[outputs])

def make_multi_conv_classifier(codings_size, shape, filter_1d_size=11, filter_num = 4, kernel_2d_col = 3, class_num = 10):
    '''
    Function classifier that will be the baseline that does not utilize weight sharing for calculating distances.
    '''           
    inputs = keras.layers.Input(shape=shape)
    z = keras.layers.Masking(mask_value=0)(inputs)
    
    # Block 1: 2D
    z = keras.layers.Conv2D(filters=filter_num, kernel_size=(4,kernel_2d_col), use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)
    z = keras.layers.Reshape((shape[1]-(kernel_2d_col - 1), filter_num))(z)
    
    z = keras.layers.Conv1D(filters=filter_num * 2, kernel_size=filter_1d_size,  use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)
    
    z = keras.layers.Conv1D(filters=filter_num * 4, kernel_size=filter_1d_size,  use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)
    
    z = keras.layers.MaxPooling1D(pool_size=2)(z)
    
    # Block 2: 1D
    z = keras.layers.Conv1D(filters=filter_num*2, kernel_size=filter_1d_size,  use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)
    z = keras.layers.MaxPooling1D(pool_size=2)(z)
    
    # Block 3: 1D
    z = keras.layers.Conv1D(filters=filter_num*4, kernel_size=filter_1d_size, use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)
    z = keras.layers.MaxPooling1D(pool_size=2)(z)
    
    # Block 4: 1D 
    z = keras.layers.Conv1D(filters=filter_num*8, kernel_size=filter_1d_size, use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)
    z = keras.layers.MaxPooling1D(pool_size=2)(z)
    
    # Block 5: 1D
    z = keras.layers.Conv1D(filters=filter_num*16, kernel_size=filter_1d_size, use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)
    g_max_pooling = keras.layers.GlobalMaxPooling1D()(z)
        
    # Output branch (binary): Dense
    binary = keras.layers.Dense(codings_size, use_bias = False)(g_max_pooling)
    binary = keras.layers.BatchNormalization()(binary)
    binary = keras.layers.Activation(activation='selu')(binary)
    output_2 = keras.layers.Dense(1, activation='sigmoid', name='output_2')(binary)
    
    # Output branch (multi-label): Dense
    multi = keras.layers.Dense(codings_size, use_bias=False)(binary)
    multi = keras.layers.BatchNormalization()(multi)
    multi = keras.layers.Activation(activation='selu')(multi)
    output_1 = keras.layers.Dense(class_num, activation='sigmoid', name='output_1')(multi)
    
    return keras.Model(inputs=inputs, outputs=[output_1, output_2])

def crm_specificity(y_true, y_pred):
    """
    param:
    y_pred - Predicted labels
    y_true - True labels 
    Returns:
    Specificity score
    """
    # assert len(y_true) == len(y_pred), f"Length disagreement: {len(y_true)} vs. {len(y_pred)}"

    # prediction may be sigmoid so round to 0 and 1 then cast to int32
    y_pred = tf.cast(tf.math.round(y_pred), tf.int32)
    
    n_total = 0.0
    n_correct = 0.0
    
    for i in range(len(y_true)):
        if y_true[i] == 0:
            n_total += 1
            if y_pred[i] == 0:
                n_correct += 1
    
    s = n_correct / (n_total + sys.float_info.epsilon)
    return s


def crm_recall(y_true, y_pred):
    """
    returns the recall; true positives to the number of total positives
    y_true: vector of true labels
    y_pred: vector of predicted labels
    """
    # assert len(y_true) == len(y_pred), f"Length disagreement: {len(y_true)} vs. {len(y_pred)}"

    # prediction may be sigmoid so round to 0 and 1 then cast to int32
    y_pred = tf.cast(tf.math.round(y_pred), tf.int32)
    
    tp = 0.0
    total = 0.0
    
    for i in range(len(y_true)):
        if y_true[i] == 1:
            total += 1
            if y_pred[i] == 1:
                tp += 1
                
    return tp / (total + sys.float_info.epsilon)

def crm_precision(y_true, y_pred):
    """
    returns the precision; true positives to the number of predicted positives
    y_true: vector of true labels
    y_pred: vector of predicted labels
    """
    
    # assert len(y_true) == len(y_pred), f"Length disagreement: {len(y_true)} vs. {len(y_pred)}"
    
    # prediction may be sigmoid so round to 0 and 1 then cast to int32
    y_pred = tf.cast(tf.math.round(y_pred), tf.int32)
    
    fp = 0.0
    tp = 0.0
    
    for i in range(len(y_true)):
        if y_pred[i] == 1:
            if y_true[i] == 1:
                tp += 1
            else:
                fp += 1
                
    return tp / (tp + fp + sys.float_info.epsilon)

def crm_f1_score(y_true, y_pred):
    """
    returns the f1_score; the harmonic mean between recall and precision
    y_true: vector of true labels
    y_pred: vector of predicted labels
    """
    # assert len(y_true) == len(y_pred), f"Length disagreement: {len(y_true)} vs. {len(y_pred)}"

    # prediction may be sigmoid so round to 0 and 1 then cast to int32
    y_pred = tf.cast(tf.math.round(y_pred), tf.int32)
    
    r = crm_recall(y_true, y_pred)
    p = crm_precision(y_true, y_pred)
    
    return (2 * r * p) / (r + p + sys.float_info.epsilon)
            

def crm_accuracy(y_true, y_pred):
    """
    returns the accuracy;
    y_true: vector of true labels
    y_pred: vector of predicted labels    
    """
    
    y_pred = tf.cast(tf.math.round(y_pred), tf.int32)
    y_true = tf.cast(y_true, tf.int32)

    c = 0.0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            c += 1
            
    return c / (len(y_true) + sys.float_info.epsilon)

def print_results(seq, model):
    metric_dict = {x:0 for x in ['loss', 'accuracy', 'crm_specificity', 'recall', 'precision', 'crm_f1_score']}
    for i in range(3):
        x = model.evaluate(seq, verbose = 1)
        metric_dict['loss'] += x[0]
        metric_dict['accuracy'] += x[1]
        metric_dict['crm_specificity'] += x[2]
        metric_dict['recall'] += x[3]
        metric_dict['precision'] += x[4]
        metric_dict['crm_f1_score'] += x[5]

    for i in metric_dict:
        metric_dict[i] /= 3
        if i != 'loss':
            metric_dict[i] *= 100
            metric_dict[i] = round(metric_dict[i], 2)

        else:
            metric_dict[i] = round(metric_dict[i], 4)
        
    print(','.join([str(x) for x in list(metric_dict.values())]))


