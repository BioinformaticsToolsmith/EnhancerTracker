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

#
# UNUSED
#
# def make_regular_conv_classifier(codings_size):
#     '''
#     Function classifier that will be the baseline that does not utilize weight sharing for calculating distances.
#     '''
#     inputs = keras.layers.Input(shape=[28,28,2])
#     z = keras.layers.Conv2D(filters=32, kernel_size=3, activation='selu')(inputs)
#     z = keras.layers.MaxPooling2D(pool_size=2)(z)
#     z = keras.layers.Conv2D(filters=64, kernel_size=3, activation='selu')(z)
#     z = keras.layers.MaxPooling2D(pool_size=2)(z)
#     z = keras.layers.Conv2D(filters=128, kernel_size=3, activation='selu')(z)
#     z = keras.layers.MaxPooling2D(pool_size=2)(z)
#     z = keras.layers.Flatten()(z)
#     z = keras.layers.Dense(codings_size, activation='selu')(z)
#     outputs = keras.layers.Dense(1, activation='sigmoid')(z)
#     return keras.Model(inputs = [inputs], outputs=[outputs])

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

def make_multi_conv_classifier(codings_size, shape, filter_1d_size=11, filter_num = 4, class_num = 10):
    '''
    Function classifier that will be the baseline that does not utilize weight sharing for calculating distances.
    '''    
    
    inputs = keras.layers.Input(shape=shape)
    z = keras.layers.Masking(mask_value=0)(inputs)
    z = keras.layers.Permute((2, 1))(z)

    
    z = keras.layers.SeparableConv1D(filters=filter_num, kernel_size=filter_1d_size, use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)
    z = keras.layers.MaxPooling1D(pool_size=2)(z)

    z = keras.layers.SeparableConv1D(filters=filter_num*2, kernel_size=filter_1d_size,  use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)
    
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
    
    z = keras.layers.Dense(codings_size, use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)
    
    outputs = keras.layers.Dense(class_num, activation='sigmoid')(z)
    return keras.Model(inputs=inputs, outputs=outputs)




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

def make_conv_classifier_blocks(codings_size, shape, filter_1d_size=11, filter_num = 4, blocks = 3):
    '''
    Function classifier that will be the baseline that does not utilize weight sharing for calculating distances.
    '''    
    assert blocks >= 1, f"blocks can not be less than 1! {blocks}"
    kernel_2d_col = 3 
    
    inputs = keras.layers.Input(shape=shape)
    z = keras.layers.Masking(mask_value=0)(inputs)
    z = keras.layers.SeparableConv2D(filters=filter_num, kernel_size=(4,kernel_2d_col), use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)

    z = keras.layers.Reshape((shape[1]-(kernel_2d_col - 1), filter_num))(z)
    z = keras.layers.MaxPooling1D(pool_size=2)(z)
    
    f = 2
    for i in range(blocks):
        

        z = keras.layers.SeparableConv1D(filters=filter_num*f, kernel_size=filter_1d_size, use_bias=False)(z)
        z = keras.layers.BatchNormalization()(z)
        z = keras.layers.Activation(activation='selu')(z)

        if i < blocks - 1:
            z = keras.layers.MaxPooling1D(pool_size=2)(z)
        elif i == blocks - 1:
            z = keras.layers.GlobalMaxPooling1D()(z)
        else:
            raise RuntimeError("How did you get here???")

        
        f *= 2
    
    z = keras.layers.Dense(codings_size, use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)
    
    outputs = keras.layers.Dense(1, activation='sigmoid')(z)
    return keras.Model(inputs = [inputs], outputs=[outputs])

# def make_conv_base(codings_size, shape, filter_num = 4):
#     '''
#     A convolutional base for the triplet or the siamese networks.
#     '''    
#     kernel_2d_col = 3 
#     filter_1d_size=11
    
#     inputs = keras.layers.Input(shape=shape)
#     z = keras.layers.Masking(mask_value=0)(inputs)
#     z = keras.layers.SeparableConv2D(filters=filter_num, kernel_size=(4,kernel_2d_col), use_bias=False)(z)
#     z = keras.layers.BatchNormalization()(z)
#     z = keras.layers.Activation(activation='selu')(z)

#     z = keras.layers.Reshape((shape[1]-(kernel_2d_col - 1), filter_num))(z)
#     z = keras.layers.MaxPooling1D(pool_size=2)(z)

#     z = keras.layers.SeparableConv1D(filters=filter_num*2, kernel_size=filter_1d_size, use_bias=False)(z)
#     z = keras.layers.BatchNormalization()(z)
#     z = keras.layers.Activation(activation='selu')(z)

#     z = keras.layers.MaxPooling1D(pool_size=2)(z)
    
#     z = keras.layers.SeparableConv1D(filters=filter_num*4, kernel_size=filter_1d_size, use_bias=False)(z)
#     z = keras.layers.BatchNormalization()(z)
#     z = keras.layers.Activation(activation='selu')(z)

#     z = keras.layers.GlobalMaxPooling1D()(z)
    
#     #z = keras.layers.Flatten()(z)
#     outputs = keras.layers.Dense(codings_size)(z)
    
#     return keras.Model(inputs = [inputs], outputs=[outputs])

def make_dense_base(codings_size, shape):
    inputs = keras.layers.Input(shape=shape)
    z = keras.layers.Flatten()(inputs)
    #z = keras.layers.Masking(mask_value=0)(z)
    z = keras.layers.Dense(100)(z)
    outputs = keras.layers.Dense(codings_size)(z)
    
    return keras.Model(inputs = [inputs], outputs=[outputs])

def make_conv_base(codings_size, shape, filter_num_1 = 64, filter_num_2 = 4, filter_1d_size=11):
    '''
    A convolutional base for the triplet or the siamese networks.
    '''    
    kernel_2d_col = 3
    
    inputs = keras.layers.Input(shape=shape)
    z = keras.layers.Masking(mask_value=0)(inputs)
    z = keras.layers.Conv2D(filters=filter_num_1, kernel_size=(4,kernel_2d_col),use_bias=False)(z) #
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)

    z = keras.layers.Reshape((shape[1]-(kernel_2d_col - 1), filter_num_1))(z)
    z = keras.layers.MaxPooling1D(pool_size=2)(z)

    z = keras.layers.Conv1D(filters=filter_num_2, kernel_size=filter_1d_size, use_bias=False)(z) #
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)
    z = keras.layers.MaxPooling1D(pool_size=2)(z)
    
    z = keras.layers.Conv1D(filters=filter_num_2*2, kernel_size=filter_1d_size, use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)
    z = keras.layers.MaxPooling1D(pool_size=2)(z)
    
    # z = keras.layers.Conv1D(filters=filter_num_2*4, kernel_size=filter_1d_size, use_bias=False)(z)
    # z = keras.layers.BatchNormalization()(z)
    # z = keras.layers.Activation(activation='selu')(z)
    # z = keras.layers.MaxPooling1D(pool_size=2)(z)
    
    # z = keras.layers.Conv1D(filters=filter_num_2*8, kernel_size=filter_1d_size, use_bias=False)(z)
    # z = keras.layers.BatchNormalization()(z)
    # z = keras.layers.Activation(activation='selu')(z)
    # z = keras.layers.MaxPooling1D(pool_size=2)(z)
    
    z = keras.layers.Conv1D(filters=filter_num_2*4, kernel_size=filter_1d_size, use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)
    
    z = keras.layers.GlobalMaxPooling1D()(z)
    
    # z = keras.layers.Flatten()(z)
    outputs = keras.layers.Dense(codings_size)(z)
    
    return keras.Model(inputs = [inputs], outputs=[outputs])

def make_conv_base_sequential(codings_size, shape, filter_num_1 = 4, filter_1d_size=11):
    kernel_2d_col = 3
    model = keras.Sequential(
        [
            layers.Input(shape=shape),
            layers.Masking(mask_value=0),
            layers.Conv2D(filters=filter_num_1, kernel_size=(4,kernel_2d_col),use_bias=False),
            layers.BatchNormalization(),
            layers.Activation(activation='selu'),
            layers.Reshape((shape[1]-(kernel_2d_col - 1), filter_num_1)),
            layers.MaxPooling1D(pool_size=2),
            
            layers.Conv1D(filters=filter_num_1*2, kernel_size=filter_1d_size, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation(activation='selu'),
            layers.MaxPooling1D(pool_size=2),
            
            layers.Conv1D(filters=filter_num_1*4, kernel_size=filter_1d_size, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation(activation='selu'),
            layers.MaxPooling1D(pool_size=2),
            
            layers.Conv1D(filters=filter_num_1*8, kernel_size=filter_1d_size, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation(activation='selu'),
            layers.MaxPooling1D(pool_size=2), 
            
#             layers.Conv1D(filters=filter_num_1*16, kernel_size=filter_1d_size, use_bias=False),
#             layers.BatchNormalization(),
#             layers.Activation(activation='selu'),
#             layers.MaxPooling1D(pool_size=2), 
            
            
#             layers.Conv1D(filters=filter_num_2*4, kernel_size=filter_1d_size, use_bias=False),
#             layers.BatchNormalization(),
#             layers.Activation(activation='selu'),

            #layers.GlobalMaxPooling1D(),
            layers.Flatten(),
            
            layers.Dense(3*codings_size, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation(activation='selu'), 
            
            layers.Dense(2*codings_size, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation(activation='selu'), 
            
            layers.Dense(codings_size)
        ]
        
    )
    return model


class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs        
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean
    
# def make_conv_base(codings_size):
#     '''
#     Credit: https://www.tensorflow.org/tutorials/generative/cvae
#     '''    
#     inputs = keras.layers.Input(shape=[28,28])
#     z = keras.layers.Reshape(target_shape=(28,28,1))(inputs)
#     z = keras.layers.Conv2D(filters=32, kernel_size=3, activation='selu')(z)
#     z = keras.layers.MaxPooling2D(pool_size=2)(z)
#     z = keras.layers.Conv2D(filters=64, kernel_size=3, activation='selu')(z)
#     z = keras.layers.MaxPooling2D(pool_size=2)(z)
#     z = keras.layers.Conv2D(filters=128, kernel_size=3, activation='selu')(z)
#     z = keras.layers.Flatten()(z)
#     codings = keras.layers.Dense(codings_size)(z)
#     return keras.Model(inputs = [inputs], outputs=[codings])

def make_encoder_vae(codings_size, shape, filter_num):
    '''
    Credit: https://www.tensorflow.org/tutorials/generative/cvae
    It returns an encoder and the dimension size needed for the decoder
    '''    
    kernel_2d_col = 3 
    filter_1d_size= 12#11
    
    inputs = keras.layers.Input(shape=shape)
    z = keras.layers.Masking(mask_value=0)(inputs)
    z = keras.layers.Conv2D(filters=filter_num, kernel_size=(4,kernel_2d_col), strides=2, use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)

    #z = keras.layers.Reshape((shape[1]-(kernel_2d_col - 1), filter_num))(z)
    _, _, hzg, _ = z.shape 
    z = keras.layers.Reshape((hzg, filter_num))(z) #shape[1]//2
    #z = keras.layers.Reshape((shape[1], filter_num))(z)
    #z = keras.layers.MaxPooling1D(pool_size=2)(z)

    z = keras.layers.Conv1D(filters=filter_num*2, kernel_size=filter_1d_size, strides=2, use_bias=False, padding='same')(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)

    #z = keras.layers.MaxPooling1D(pool_size=2)(z)
    
    z = keras.layers.Conv1D(filters=filter_num*4, kernel_size=filter_1d_size, strides=2, use_bias=False, padding='same')(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)

    _, map_from, _ = z.shape
    
    #z = keras.layers.GlobalMaxPooling1D()(z)
    z = keras.layers.Flatten()(z)
    
    codings_mean = keras.layers.Dense(codings_size)(z)
    codings_log_var = keras.layers.Dense(codings_size)(z)
    codings = Sampling()([codings_mean, codings_log_var])
    
    return keras.Model(inputs = [inputs], outputs=[codings_mean, codings_log_var, codings]), map_from

def make_encoder_pooling_vae(codings_size, shape, filter_num):
    '''
    Credit: https://www.tensorflow.org/tutorials/generative/cvae
    It returns an encoder and the dimension size needed for the decoder
    '''    
    kernel_2d_col = 3 
    filter_1d_size=11
    
    inputs = keras.layers.Input(shape=shape)
    z = keras.layers.Masking(mask_value=0)(inputs)
    z = keras.layers.Conv2D(filters=filter_num, kernel_size=(4,kernel_2d_col), use_bias=False)(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)

    _, _, hzg, _ = z.shape 
    z = keras.layers.Reshape((hzg, filter_num))(z) #shape[1]//2
    #z = keras.layers.Reshape((shape[1]-(kernel_2d_col - 1), filter_num))(z)
    #z = keras.layers.Reshape((shape[1], filter_num))(z)
    z = keras.layers.MaxPooling1D(pool_size=2)(z)

    z = keras.layers.Conv1D(filters=filter_num*2, kernel_size=filter_1d_size, use_bias=False, padding='same')(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)

    z = keras.layers.MaxPooling1D(pool_size=2)(z)
    
    z = keras.layers.Conv1D(filters=filter_num*4, kernel_size=filter_1d_size, use_bias=False, padding='same')(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation='selu')(z)

    _, map_from, _ = z.shape
    
    z = keras.layers.GlobalMaxPooling1D()(z)
    
    #z = keras.layers.Flatten()(z)
    
    codings_mean = keras.layers.Dense(codings_size)(z)
    codings_log_var = keras.layers.Dense(codings_size)(z)
    codings = Sampling()([codings_mean, codings_log_var])
    
    return keras.Model(inputs = [inputs], outputs=[codings_mean, codings_log_var, codings]), map_from

# def make_decoder(codings_size, shape, filter_num, map_from):
#     '''
#     Credit: https://www.tensorflow.org/tutorials/generative/cvae
#     '''
#     row_num, col_num, cha_num = shape
    
#     filter_1d_size=11
#     #map_from = 101 # 105
    
#     decoder_inputs = keras.layers.Input(shape=[codings_size])
#     x = keras.layers.Dense(units=map_from*filter_num*4, activation="selu")(decoder_inputs) # 105
#     x = keras.layers.Reshape(target_shape=(map_from, filter_num*4))(x) # 105
#     x = keras.layers.Conv1DTranspose(filters=filter_num*4, kernel_size=filter_1d_size, strides=2, padding='same', activation='selu')(x)
#     x = keras.layers.Conv1DTranspose(filters=filter_num*2, kernel_size=filter_1d_size, strides=2, padding='same', activation='selu')(x)
#     x = keras.layers.Conv1DTranspose(filters=filter_num,   kernel_size=filter_1d_size, strides=2, padding='same', activation='selu')(x)
#     x = keras.layers.Conv1DTranspose(filters=1, kernel_size=filter_1d_size, strides=2, padding='valid', activation='selu')(x) # 22
#     _, x_col_num, _ = x.shape

class transformer_encoder(layers.Layer):
    """
    Credit: Using the same encoder described in the Deep Learning with Python book by Francois Chollet
    """
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embeded_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads = num_heads, key_dim=embed_dim)
        self.dense_poj = keras.Sequential(
            [layers.Dense(dense_dim, activation = "relu"),
             layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(
            inputs,inputs, attention_mask = mask)
        proj_input = sef.layernorm_1(inputs + attention_output) 
        proj_output - self.dense_rpoj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        
        return config
            
class SiameseNet(keras.Model):
    '''
    Credit: We are just converting the orginal pytorch code to keras code
    Orfilter_num=16iginal code: https://github.com/hmishfaq/DDSM-TVAE/blob/master/main.py
    '''
    def __init__(self, an_encoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder  = an_encoder
        # self.distance = cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
        
    def get_encoder(self):
        return self.encoder
        
    def call(self, X):
        C_a = self.encoder(X[:, :, :, 0])
        C_b = self.encoder(X[:, :, :, 1])
        
        # D = self.distance(C_a, C_b)    
        # print(D.shape)
        
        D = tf.norm(C_a-C_b, ord='euclidean', axis=1)
        
        return D
#     leftover = (row_num * col_num) - x_col_num + 1
#     # if leftover > 0:
#     x = keras.layers.Conv1DTranspose(filters=1, kernel_size=leftover, strides=1, padding='valid', activation='sigmoid')(x)
    
#     outputs = keras.layers.Reshape(target_shape=shape)(x)
#     return keras.Model(inputs=[decoder_inputs], outputs=[outputs])
#     #return keras.Model(inputs=[decoder_inputs], outputs=[x])

def make_decoder(codings_size, shape, filter_num, map_from):
    '''
    Credit: https://www.tensorflow.org/tutorials/generative/cvae
    '''
    row_num, col_num, cha_num = shape
    
    filter_1d_size= 12

    
    decoder_inputs = keras.layers.Input(shape=[codings_size])
    x = keras.layers.Dense(units=map_from*filter_num*4, activation="selu")(decoder_inputs) # 105
    x = keras.layers.Reshape(target_shape=(map_from, filter_num*4))(x) # 105
    x = keras.layers.Conv1DTranspose(filters=filter_num*4, kernel_size=filter_1d_size, strides=2, padding='same', activation='selu')(x)
    x = keras.layers.Conv1DTranspose(filters=filter_num*2, kernel_size=filter_1d_size, strides=2, padding='same', activation='selu')(x)
    x = keras.layers.Conv1DTranspose(filters=filter_num,   kernel_size=filter_1d_size, strides=2, padding='same', activation='selu')(x)
    x = keras.layers.Conv1DTranspose(filters=1, kernel_size=filter_1d_size, strides=2, padding='valid', activation='sigmoid')(x)
    
    _, x_col_num, _ = x.shape
    leftover = (row_num * col_num) - x_col_num + 1
    if leftover > 0:
        x = keras.layers.Conv1DTranspose(filters=1, kernel_size=leftover, strides=1, padding='valid', activation='sigmoid')(x)
        
    if x_col_num - (row_num * col_num) == 2:
        x = keras.layers.Conv1D(filters=1, kernel_size=3, strides=1, padding='valid', activation='sigmoid')(x)
    
    outputs = keras.layers.Reshape(target_shape=shape)(x)
    
    return keras.Model(inputs=[decoder_inputs], outputs=[outputs])
    #return keras.Model(inputs=[decoder_inputs], outputs=[x])

class VAE(keras.Model):
    def __init__(self, an_encoder, a_decoder,  **kwargs):
        super().__init__(**kwargs)
        self.encoder = an_encoder
        self.decoder = a_decoder
        
    def call(self, X):
        M, V, C = self.encoder(X)       
                        
        # Ran into an error when it came to the stacking of the decoder where the channel was 
        # placed right after the batch size instead of at the end
        X_hat = self.decoder(C)
        
        M_V   = tf.stack([M, V], axis=2)
        
        return {'recon':X_hat, 'mean-var':M_V}


    
class TripletNetVAE(keras.Model):
    '''
    Credit: We are just converting the orginal pytorch code to keras code
    Original code: https://github.com/hmishfaq/DDSM-TVAE/blob/master/main.py
    '''
    def __init__(self, an_encoder, a_decoder, distance_func, **kwargs):
        
        super().__init__(**kwargs)
        self.encoder = an_encoder
        self.decoder = a_decoder
        self.distance_func = distance_func

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
    
    def get_distance_func(self):
        return self.distance_func
        
    def call(self, X):
        '''
        X[:, 0]: is the anchor matrix
        X[:, 1]: is the positive matrix
        X[:, 2]: is the negative matrix
        '''                    

        M_a, V_a, C_a = self.encoder(X[:, :, :, 0])        
        M_p, V_p, C_p = self.encoder(X[:, :, :, 1])
        M_n, V_n, C_n = self.encoder(X[:, :, :, 2])
            
        X_hat = tf.stack([self.decoder(C_a), self.decoder(C_p), self.decoder(C_n)], axis=3) 

        M_V   = tf.stack([M_a, V_a, M_p, V_p, M_n, V_n], axis=2)
        
        D = self.distance_func(M_V)

        return {'recon':X_hat, 'mean-var':M_V, 'distance':D}

class TripletNet(keras.Model):
    '''
    Credit: We are just converting the orginal pytorch code to keras code
    Original code: https://github.com/hmishfaq/DDSM-TVAE/blob/master/main.py
    '''
    def __init__(self, an_encoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = an_encoder

    def get_encoder(self):
        return self.encoder
        
    def call(self, X):
        C_a = self.encoder(X[:, :, :, 0])        
        C_p = self.encoder(X[:, :, :, 1])
        C_n = self.encoder(X[:, :, :, 2])
        
        C_a = C_a / tf.norm(C_a, ord='euclidean', axis=1, keepdims=True)        
        C_p = C_p / tf.norm(C_p, ord='euclidean', axis=1, keepdims=True)
        C_n = C_n / tf.norm(C_n, ord='euclidean', axis=1, keepdims=True)
                
        D_p = tf.norm(C_a-C_p, ord='euclidean', axis=1)
        D_n = tf.norm(C_a-C_n, ord='euclidean', axis=1)
        D   = tf.stack([D_p, D_n], axis = 1)
        
        return D
    
    
class SiameseNet(keras.Model):
    '''
    Credit: We are just converting the orginal pytorch code to keras code
    Orfilter_num=16iginal code: https://github.com/hmishfaq/DDSM-TVAE/blob/master/main.py
    '''
    def __init__(self, an_encoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder  = an_encoder
        # self.distance = cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
        
    def get_encoder(self):
        return self.encoder
        
    def call(self, X):
        C_a = self.encoder(X[:, :, :, 0])
        C_b = self.encoder(X[:, :, :, 1])
        
        C_a = C_a / tf.norm(C_a, ord='euclidean', axis=1, keepdims=True)        
        C_b = C_b / tf.norm(C_b, ord='euclidean', axis=1, keepdims=True)
        
        # D = self.distance(C_a, C_b)    
        # print(D.shape)
        
        D = tf.norm(C_a-C_b, ord='euclidean', axis=1)
        
        return D
    
class SiameseNetVAE(keras.Model):
    '''
    Authors: Anthony B Garza
             Rolando Garcia
             Hani Z Girgis
    The siamese network combined with a variational auto encoder.
    '''
    def __init__(self, an_encoder, a_decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = an_encoder
        self.decoder = a_decoder

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
           
    def call(self, X):
        M_1, V_1, C_1 = self.encoder(X[:, :, :, 0])        
        M_2, V_2, C_2 = self.encoder(X[:, :, :, 1])
            
        X_hat = tf.stack([self.decoder(C_1), self.decoder(C_2)], axis=3) 

        M_V   = tf.stack([M_1, V_1, M_2, V_2], axis=2)
        
        D = tf.norm(M_1-M_2, ord='euclidean', axis=1)
        
        return {'recon': X_hat, 'mean-var': M_V, 'distance': D}
          
class AngularNet(keras.Model):
    def __init__(self, an_encoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = an_encoder

    def get_encoder(self):
        return self.encoder
        
    def call(self, X):
        # Normalize each row to a vector on size 1
        C_a = self.encoder(X[:, :, :, 0]) 
        C_p = self.encoder(X[:, :, :, 1]) 
        C_n = self.encoder(X[:, :, :, 2])
        
        # C_a = C_a / tf.norm(C_a, ord='euclidean', axis=1, keepdims=True)        
        # C_p = C_p / tf.norm(C_p, ord='euclidean', axis=1, keepdims=True)
        # C_n = C_n / tf.norm(C_n, ord='euclidean', axis=1, keepdims=True)
        
        # print(tf.norm(C_a, ord='euclidean', axis=1).shape)
        
        C_c = (C_a + C_p) / 2.0
        
        D_p = tf.norm(C_a-C_p, ord='euclidean', axis=1)
        D_n = tf.norm(C_n-C_c, ord='euclidean', axis=1)
        D   = tf.stack([D_p, D_n], axis = 1)
        
        return D    
    
class ClassifierAngularTriplet(keras.Model):
    '''
    A classifier for the encoding learned by the angular loss
    '''
    def __init__(self, an_encoder, codings_size, dense_size, **kwargs):
        super().__init__(**kwargs)
        self.encoder    = an_encoder
        
        # Freeze the encoder
        for layer in self.encoder.layers:
            layer.trainable = False
        
        self.sequential = keras.Sequential([
            keras.Input(shape=(2,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(dense_size, activation="selu"),
            keras.layers.Dense(1, activation="sigmoid")
        ]
        )
        
    def get_encoder(self):
        return self.encoder
        
    def call(self, X):
        C_a = self.encoder(X[:, :, :, 0]) 
        C_p = self.encoder(X[:, :, :, 1]) 
        C_n = self.encoder(X[:, :, :, 2])
                
        C_c = (C_a + C_p) / 2.0
        
        D_p = tf.norm(C_a-C_p, ord='euclidean', axis=1)
        D_n = tf.norm(C_n-C_c, ord='euclidean', axis=1)
        D   = tf.stack([D_p, D_n], axis = 1)
        
        return self.sequential(D)    

class ClassifierDistanceTriplet(keras.Model):
    '''
    A classifier for the encoding learned by the triplet loss
    '''
    def __init__(self, an_encoder, codings_size, dense_size, is_vae=False, **kwargs):
        super().__init__(**kwargs)
        self.encoder = an_encoder
        self.is_vae  = is_vae
        
        # Freeze the encoder
        for layer in self.encoder.layers:
            layer.trainable = False
        
        self.sequential = keras.Sequential([
            keras.Input(shape=(2,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(dense_size, activation="selu"),
            keras.layers.Dense(1, activation="sigmoid")
        ]
        )
        
    def get_encoder(self):
        return self.encoder
        
    def call(self, X):
        if self.is_vae:
            C_a = self.encoder(X[:, :, :, 0])[0]        
            C_p = self.encoder(X[:, :, :, 1])[0]
            C_n = self.encoder(X[:, :, :, 2])[0]
        else:
            C_a = self.encoder(X[:, :, :, 0])        
            C_p = self.encoder(X[:, :, :, 1])
            C_n = self.encoder(X[:, :, :, 2])
                    
        D_p = tf.norm(C_a-C_p, ord='euclidean', axis=1)
        D_n = tf.norm(C_a-C_n, ord='euclidean', axis=1)
        D   = tf.stack([D_p, D_n], axis = 1)
        
        return self.sequential(D)   


class ClassifierTriplet(keras.Model):
    '''
    Credit: We are just converting the orginal pytorch code to keras code
    Orfilter_num=16iginal code: https://github.com/hmishfaq/DDSM-TVAE/blob/master/main.py
    '''
    def __init__(self, an_encoder, codings_size, dense_size, **kwargs):
        super().__init__(**kwargs)
        self.encoder    = an_encoder
        
        # Freeze the encoder
        for layer in self.encoder.layers:
            layer.trainable = False
        
        self.sequential = keras.Sequential([
            keras.Input(shape=(3*codings_size,)),
            keras.layers.Dense(dense_size, activation="selu"),
            keras.layers.Dense(dense_size, activation="selu"),
            keras.layers.Dense(1, activation="sigmoid")
        ]
        )
        
    def get_encoder(self):
        return self.encoder
        
    def call(self, X):
        C_1 = self.encoder(X[:, :, :, 0])
        C_2 = self.encoder(X[:, :, :, 1])
        C_3 = self.encoder(X[:, :, :, 2])
        C = tf.concat([C_1, C_2, C_3], 1)
        return self.sequential(C)
        
class ClassifierPair(keras.Model):
    '''
    Credit: We are just converting the orginal pytorch code to keras code
    Orfilter_num=16iginal code: https://github.com/hmishfaq/DDSM-TVAE/blob/master/main.py
    '''
    def __init__(self, an_encoder, codings_size, dense_size, **kwargs):
        super().__init__(**kwargs)
        
        self.encoder    = an_encoder
        # Freeze the encoder
        for layer in self.encoder.layers:
            layer.trainable = False
        
        self.sequential = keras.Sequential([
            keras.Input(shape=(2*codings_size,)),
            keras.layers.Dense(dense_size, activation="selu"),
            keras.layers.Dense(dense_size, activation="selu"),
            #keras.layers.BatchNormalization(),
            keras.layers.Dense(1, activation="sigmoid")
        ]
        )
        
    def get_encoder(self):
        return self.encoder
        
    def call(self, X):
        C_1 = self.encoder(X[:, :, :, 0])
        C_2 = self.encoder(X[:, :, :, 1])

        C = tf.concat([C_1, C_2], 1)
        return self.sequential(C)
        

class SiameseNetClassifier(keras.Model):
    '''
    Credit: We are just converting the orginal pytorch code to keras code
    Original code: https://github.com/hmishfaq/DDSM-TVAE/blob/master/main.py
    '''
    def __init__(self, an_encoder, **kwargs):
        
        super().__init__(**kwargs)
        self.encoder = an_encoder

    def get_encoder(self):
        return self.encoder
        
    def call(self, X):

        C_a = self.encoder(X[:, :, :, 0])
        C_b = self.encoder(X[:, :, :, 1])
            
        concat = layers.Concatenate()([C_a, C_b])
        l = layers.Dense(40, activation = 'selu')(concat)
        out = layers.Dense(1, activation ='sigmoid')(l)
        
        return out

################################################################################
# Distance Functions
################################################################################
def ecludean_distance(M_V):
    M_a = M_V[:, :, 0]
    M_p = M_V[:, :, 2]
    M_n = M_V[:, :, 4]
    D_p = tf.norm(M_a-M_p, ord='euclidean', axis=1)
    D_n = tf.norm(M_a-M_n, ord='euclidean', axis=1)
    D = tf.stack([D_p, D_n], axis = 1)
    
    return D

def mahalanobis(x, mean, inv_cov):
    x_centered = x - mean
    r = tf.sqrt(tf.reduce_sum(tf.matmul(x_centered, inv_cov) * x_centered, axis=-1))
    return r

def manahabolis_distance(M_V):
    M_a = M_V[:, :, 0]
    V_a = M_V[:, :, 1]
    M_p = M_V[:, :, 2]
    V_p = M_V[:, :, 3]
    M_n = M_V[:, :, 4]
    V_n = M_V[:, :, 5]
    
    codings_size = M_a.shape[1]
    cov = tf.zeros((codings_size, codings_size))
    for i in range(codings_size):
        cov = tf.tensor_scatter_nd_update(cov, [[i, i]], [tf.reduce_mean(V_a[:, i] + V_p[:, i] + V_n[:, i]) / 3])
    inv_cov = tf.linalg.inv(cov)
    # cov = (V_a + V_p + V_n) / 3
    # cov_inv = tf.linalg.inv(cov)
    D_p = mahalanobis(M_a, M_p, inv_cov)
    D_n = mahalanobis(M_a, M_n, inv_cov)
    D = tf.stack([D_p, D_n], axis = 1)
    
    return D


################################################################################
# Loss functions
# Each loss function returns a tensorflow compatible loss function with certain
# parameters set from arguments to the outer function.
################################################################################

def get_triplet_loss(alpha = 0.5):   
    def triplet_loss(_, D):

        basic = D[:, 0] - D[:, 1] + alpha
        return tf.reduce_mean(tf.maximum(basic, 0.))
    return triplet_loss

def get_kld_loss(image_count = 3):
    m_list = [x * 2 for x in range(image_count)]
    v_list = [x + 1 for x in m_list]
    def kld_loss(_, M_V):    
        M = tf.gather(M_V, indices=m_list, axis=2)
        V = tf.gather(M_V, indices=v_list, axis=2)

        latent_loss = -0.5 * K.sum(1 + V - K.exp(V) - K.square(M), axis = -1)
        return K.mean(latent_loss) / (image_count * 784)
    
    return kld_loss

def get_recon_loss():
    def recon_loss(y_true, y_hat):
        return tf.reduce_mean(tf.square(y_hat - y_true))
    return recon_loss

def get_contrastive_loss(alpha = 0.5):
    #
    # Credit: Based on code from https://github.com/tensorflow/addons/blob/v0.17.0/tensorflow_addons/losses/contrastive.py#L72-L120
    #    
    def contrastive_loss(y, d):
        y = tf.cast(y, tf.float32)    
        return y * K.square(d) + (1.0 - y) * K.square(K.maximum(alpha - d, 0.0))
    return contrastive_loss

#
# The definition of alpha and sq_tan_alpha are from: https://github.com/geonm/tf_angular_loss/blob/master/tf_angular_loss.py
#
def get_angular_loss(degree):   
    # @tf.function
    def angular_loss(_, D):
        alpha = np.deg2rad(degree)
        sq_tan_alpha = np.tan(alpha) ** 2
        
        basic = tf.square(D[:, 0]) - 4 * sq_tan_alpha * tf.square(D[:, 1])

        return tf.reduce_mean(tf.maximum(basic, 0.))
    return angular_loss

################################################################################
# Evaluation metrics
################################################################################
class TripletAccuracy(keras.metrics.Metric):
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
        self.alpha = alpha
        
    def update_state(self, _, D, sample_weight=None):
        correct = D[:, 0] + self.alpha < D[:, 1]
        self.total.assign_add( len(np.where(correct == True)[0]) )
        self.count.assign_add( tf.cast(len(D), tf.float32) )
        
        assert self.total <= self.count, f'{D.shape}'
    
    def result(self):
        assert self.total <= self.count
        return self.total / self.count
    
class PairAccuracy(keras.metrics.Metric):
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
        self.alpha = alpha
        
    def update_state(self, y, D, sample_weight=None):
        y = tf.squeeze(y)
        # D = tf.cast(D, tf.float64)

        y_np = y.numpy()
        D_np = D.numpy()
        
        correct = 0
        for i in range(len(y_np)):
            if (y_np[i] == 1 and D_np[i] < self.alpha) or (y_np[i] == 0 and D_np[i] >= self.alpha):
                correct += 1
                
        # correct = tf.reduce_sum(tf.cast(tf.equal(y, True) & (D < self.alpha), tf.int64)) + tf.reduce_sum(tf.cast(tf.equal(y, False) & (D >= self.alpha), tf.int64))
        assert correct <= tf.size(y)

        self.total.assign_add( tf.cast(correct, tf.float32) )
        self.count.assign_add( tf.cast(tf.size(D), tf.float32) )



    
    def result(self):
        # print("\n", self.total, self.count, "\n")
        return self.total / self.count
    
    def reset_states(self):
        # Reset the metric state at the start of each epoch or batch
        K.batch_set_value([(v, 0) for v in self.variables])
 

# Works on three columns: anchor, positive, and negative or another positive
#
class AngularAccuracy(keras.metrics.Metric):
    def __init__(self, degree, **kwargs):
        super().__init__(**kwargs)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
        self.tan_alpha = np.tan(np.deg2rad(degree))
        
    def update_state(self, _, D, sample_weight=None):      #y  
        correct = tf.square(D[:, 0]) / (2 * tf.square(D[:, 1])) <= self.tan_alpha #dissimilar in place of correct
        # ground = np.squeeze(y, axis=1) == 0 
        #correct = dissimilar # == ground
        
        total = len(np.where(correct == True)[0])
        count = tf.cast(len(D), tf.float32)
        
        #print(dissimilar.shape, ground.shape, correct.shape, total, count)
        
        self.total.assign_add( total )
        self.count.assign_add( count )
        
        assert self.total <= self.count, f'{total}, {count}'
    
    def result(self):
        assert self.total <= self.count
        return self.total / self.count

#
# Works on three columns: Anchor, Positive, and Negative
#
# class AngularAccuracy(keras.metrics.Metric):
#     def __init__(self, degree, **kwargs):
#         super().__init__(**kwargs)
#         self.total = self.add_weight("total", initializer="zeros")
#         self.count = self.add_weight("count", initializer="zeros")
#         self.tan_alpha = np.tan(np.deg2rad(degree))
        
#     def update_state(self, _, D, sample_weight=None):
#         #correct = D[:, 0] + self.alpha < D[:, 1]
        
#         correct = tf.square(D[:, 0]) / (2 * tf.square(D[:, 1])) <= self.tan_alpha 
        
#         self.total.assign_add( len(np.where(correct == True)[0]) )
#         self.count.assign_add( tf.cast(len(D), tf.float32) )
        
#         assert self.total <= self.count, f'{D.shape}'
    
#     def result(self):
#         assert self.total <= self.count
#         return self.total / self.count
    
def evaluate_encoder_on_pair(a_pair_seq, an_encoder, alpha=0.5, is_vae = True):
    '''
    a_pair_seq: A pair seq
    an_encoder: A trained encoder
    alpha: Is the threshold for determining if the pair are similar according to the absolute objective of the pair accuracy
    '''
    pair_acc = PairAccuracy(alpha)
    
    for x_batch, y_batch in a_pair_seq:
        print('.', end='')
        #
        # Evaluates a pair with an encoder by calculating the ecludian distance 
        # of a pair and comparing to a threshold alpha
        #
        
        a = an_encoder.predict(x_batch[:, :, :, 0], verbose = 0)
        b = an_encoder.predict(x_batch[:, :, :, 1], verbose = 0)
        
        if is_vae:
            a = a[0]
            b = b[0]
        
        d = tf.norm(a-b, ord='euclidean', axis=1)
                    
        pair_acc.update_state(y_batch, d) 
    print('')    

    print(pair_acc.result().numpy())


def evaluate_siamese_on_triplets(a_triplet_seq, a_siamese_net, alpha=0.5):
    '''
    a_triplet_seq: A triplet seq
    a_siamese_net: A trained siamese network
    alpha: Is the threshold used in: correct = D[:, 0] + self.alpha < D[:, 1]
    '''
    triplet_acc = TripletAccuracy(alpha)
    
    for x_batch, y_batch in a_triplet_seq:
        print('.', end='')
        #
        # Evaluates a triplet with the siamese net by comparing the ecludian distance 
        # of the anchor to the positive and the ecludian distance of the anchor to the 
        # negative
        #
        
        D_p = a_siamese_net.predict(x_batch[:, :, :, [0,1]], verbose = 0)
        D_n = a_siamese_net.predict(x_batch[:, :, :, [0,2]], verbose = 0)
                
        D = tf.stack([D_p, D_n], axis = 1)
        
        triplet_acc.update_state(None, D) 
    print('')    

    print(triplet_acc.result().numpy())
    
def evaluate_vae_on_triplets(a_triplet_seq, a_vae_net, alpha=0.5):
    '''
    a_triplet_seq: A triplet seq
    a_vae_net: A trained vae network
    alpha: Is the threshold used in: correct = D[:, 0] + self.alpha < D[:, 1]
    '''
    triplet_acc = TripletAccuracy(alpha)
    
    for x_batch, y_batch in a_triplet_seq:
        print('.', end='')
        #
        # Evaluates a triplet with the vae  net by comparing the ecludian distance 
        # of the anchor to the positive and the ecludian distance of the anchor to the 
        # negative
        #
        a = a_vae_net.encoder.predict(x_batch[:, :, :, [0]], verbose = 0)[2]
        p = a_vae_net.encoder.predict(x_batch[:, :, :, [1]], verbose = 0)[2]
        n = a_vae_net.encoder.predict(x_batch[:, :, :, [2]], verbose = 0)[2]
        D_p = tf.norm(a-p, ord='euclidean', axis=1)
        D_n = tf.norm(a-n, ord='euclidean', axis=1)
                
        D = tf.stack([D_p, D_n], axis = 1)
        
        triplet_acc.update_state(None, D) 
    print('')    

    print(triplet_acc.result().numpy())
    
def evaluate_siamese_vae_on_triplets(a_triplet_seq, a_siamese_net, alpha=0.5):
    '''
    a_triplet_seq: A triplet seq
    a_siamese_net: A trained siamese network
    alpha: Is the threshold used in: correct = D[:, 0] + self.alpha < D[:, 1]
    '''
    triplet_acc = TripletAccuracy(alpha)
    
    for x_batch, y_batch in a_triplet_seq:
        print('.', end='')
        #
        # Evaluates a triplet with the siamese net by comparing the ecludian distance 
        # of the anchor to the positive and the ecludian distance of the anchor to the 
        # negative
        #
        
        D_p = a_siamese_net.predict(x_batch[:, :, :, [0,1]], verbose = 0)['distance']
        D_n = a_siamese_net.predict(x_batch[:, :, :, [0,2]], verbose = 0)['distance']
                
        D = tf.stack([D_p, D_n], axis = 1)
        
        triplet_acc.update_state(None, D) 
    print('')    

    print(triplet_acc.result().numpy())

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
            
##############################################################################
# Optimizers
##############################################################################
def get_nesterov(lr = 0.1, momentum = 0.9):
    return keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)

#############################################################################
# Schedulers
#############################################################################
def exponential_decay_fn(epoch, lr):
    return lr*0.1**(1/20)

scheduler_dict = {"exponential_decay_fn": exponential_decay_fn}

def get_lr_scheduler(name = "exponential_decay_fn"):
    return keras.callbacks.LearningRateScheduler(scheduler_dict[name])

#############################################################################
# Callbacks
#############################################################################
def get_early_stopping(monitor, patience=20, min_delta=1/100000, restore_best_weights = True):
    return keras.callbacks.EarlyStopping(patience=patience, min_delta=min_delta, restore_best_weights=restore_best_weights, monitor=monitor)

if __name__ == "__main__":
    sm = make_siamese_net(10)
    sm.get_encoder().summary()
