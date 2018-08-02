from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, GlobalAveragePooling3D,\
                         GlobalMaxPooling3D, AveragePooling3D, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,\
                         GlobalMaxPooling2D, AveragePooling2D, Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape, Activation
from keras.layers.merge import Concatenate, add
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import keras.backend as K

from .multi_gpu import ModelMGPU
import json

def phinet_2D(n_classes, model_path, num_channels=1, learning_rate=1e-3, num_gpus=1):

    inputs = Input(shape=(None,None,num_channels))

    x = Conv2D(8, (3,3), strides=(2,2), padding='same')(inputs)
    x = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x)

    x = Conv2D(16, (3,3), strides=(2,2), padding='same')(x)
    x = BatchNormalization()(x)
    y = Activation('relu')(x)
    x = Conv2D(16, (3,3), strides=(1,1), padding='same')(y)
    x = BatchNormalization()(x)
    x = add([x, y])
    x = Activation('relu')(x)

    # this block will pool a handful of times to get the "big picture" 
    y = MaxPooling2D(pool_size=(5,5), strides=(2,2), padding='same')(inputs)
    y = AveragePooling2D(pool_size=(3,3), strides=(2,2), padding='same')(y)
    y = Conv2D(16, (3,3), strides=(1,1), padding='same')(y)

    # this layer will preserve original signal
    z = Conv2D(8, (3,3), strides=(2,2), padding='same')(inputs)
    z = Conv2D(12, (3,3), strides=(2,2), padding='same')(z)
    z = Conv2D(16, (3,3), strides=(1,1), padding='same')(z)

    x = Concatenate(axis=-1)([x, y, z])

    # global avg pooling before FC
    x = GlobalAveragePooling2D()(x)
    x = Dense(n_classes)(x)

    pred = Activation('softmax')(x)
    
    model = Model(inputs=inputs, outputs=pred)

    model.compile(optimizer=Adam(lr=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # save json before checking if multi-gpu
    json_string = model.to_json()
    with open(model_path, 'w') as f:
        json.dump(json_string, f)

    print(model.summary())

    # recompile if multi-gpu model
    if num_gpus > 1:
        model = ModelMGPU(model, num_gpus)
        model.compile(optimizer=Adam(lr=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
    return model

def phinet(n_classes, model_path, num_channels=1, learning_rate=1e-3, num_gpus=1):
    inputs = Input(shape=(None,None,None,num_channels))

    x = Conv3D(8, (3,3,3), strides=(2,2,2), padding='same')(inputs)
    x = MaxPooling3D(pool_size=(3,3,3), strides=(1,1,1), padding='same')(x)

    x = Conv3D(16, (3,3,3), strides=(2,2,2), padding='same')(x)
    x = BatchNormalization()(x)
    y = Activation('relu')(x)
    x = Conv3D(16, (3,3,3), strides=(1,1,1), padding='same')(y)
    x = BatchNormalization()(x)
    x = add([x, y])
    x = Activation('relu')(x)

    # this block will pool a handful of times to get the "big picture" 
    y = MaxPooling3D(pool_size=(5,5,5), strides=(2,2,2), padding='same')(inputs)
    y = AveragePooling3D(pool_size=(3,3,3), strides=(2,2,2), padding='same')(y)
    y = Conv3D(16, (3,3,3), strides=(1,1,1), padding='same')(y)

    # this layer will preserve original signal
    z = Conv3D(8, (3,3,3), strides=(2,2,2), padding='same')(inputs)
    z = Conv3D(12, (3,3,3), strides=(2,2,2), padding='same')(z)
    z = Conv3D(16, (3,3,3), strides=(1,1,1), padding='same')(z)

    x = Concatenate(axis=4)([x, y, z])

    # global avg pooling before FC
    x = GlobalAveragePooling3D()(x)
    x = Dense(n_classes)(x)

    pred = Activation('softmax')(x)
    
    model = Model(inputs=inputs, outputs=pred)

    model.compile(optimizer=Adam(lr=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    # save json before checking if multi-gpu
    json_string = model.to_json()
    with open(model_path, 'w') as f:
        json.dump(json_string, f)

    print(model.summary())

    # recompile if multi-gpu model
    if num_gpus > 1:
        model = ModelMGPU(model, num_gpus)
        model.compile(optimizer=Adam(lr=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    return model
