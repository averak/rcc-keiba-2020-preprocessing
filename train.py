#!/usr/bin/env python
import numpy as np
import nnet
import preprocessing

x = preprocessing.FEATURE[:, :8]
y = preprocessing.FEATURE[:, 8]

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
