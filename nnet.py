import tensorflow.keras.layers as layers
from tensorflow.keras import Model


def nnet(input_shape):
    # 入力層
    input_layer = layers.Input(shape=input_shape)

    # 隠れ層
    x1 = layers.Dense(16, activation='relu')(input_layer)
    x1 = layers.Dropout(0.1)(x1)
    x2 = layers.Dense(16, activation='relu')(x1)
    x2 = layers.Dropout(0.4)(x2)

    # 出力層
    y = layers.Dense(1, activation='sigmoid')(x2)

    return Model(inputs=input_layer, outputs=y)

