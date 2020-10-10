#!/usr/bin/env python
import numpy as np
import nnet
import csv

x = []
y = []

with open('weather.csv', 'r') as f:
    reader = csv.reader(f)

    for row in list(reader)[1:]:
        label = int(row[9] == '有')

        row.pop(0)
        row.pop(-1)
        row.pop(-1)

        x.append(row)
        y.append(label)

x = np.array(x, dtype=np.float32)
y = np.array(y)

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
    epochs=5,
    validation_split=0.1,
)
