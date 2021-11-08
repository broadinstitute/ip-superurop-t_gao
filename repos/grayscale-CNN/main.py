import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses

# TODO: load and split train/test/validation data

def build_model(layers):
    model = models.Sequential()
    for layer in layers:
        model.add(layer)
    return model

if __name__ == '__main__':
    alexnet = build_model([
        layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear", input_shape=x_train.shape[1:])

        layers.Conv2D(96, 11, strides=4, padding='same')
        layers.Lambda(tf.nn.local_response_normalization)
        layers.Activation('relu')
        layers.MaxPooling2D(3, strides=2)

        layers.Conv2D(384, 3, strides=4, padding='same')
        layers.Activation('relu')

        layers.Conv2D(384, 3, strides=4, padding='same')
        layers.Activation('relu')

        layers.Flatten()
        layers.Dense(4096, activation='relu')
        layers.Dropout(0.5)

        layers.Dense(10, activation='softmax')
    ])

    print(alexnet.summary())

    alexnet.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    # TODO: train/test
    # TODO: validate
