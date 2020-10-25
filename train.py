#!/usr/bin/env python
import numpy as np
import nnet
import preprocessing

x = preprocessing.X
y = preprocessing.Y

model = nnet.nnet(x.shape[1:])
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

model.fit(
    x,
    y,
    batch_size=32,
    epochs=10,
    validation_split=0.1,
)
