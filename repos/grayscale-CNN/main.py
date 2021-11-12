import pathlib
import PIL
import PIL.Image

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import datasets, layers, models, losses

# set parameters
seed = 123
image_dir = '../../../Greyscale_Images/'
img_height = 2048
img_width = 2048
alexnet_layers = [
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
]

def build_model(layers, print_summary=False):
    model = models.Sequential()
    for layer in layers:
        model.add(layer)

    if print_summary:
        print(alexnet.summary())

    return model

if __name__ == '__main__':
    # load datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='training',
        seed=seed,
        image_size=(img_height, img_width),
    )
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='validation',
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # build AlexNet CNN
    alexnet = build_model(alexnet_layers, print_summary=True)
    alexnet.compile(
        optimizer='adam',
        loss=losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )

    # create callback to save model
    checkpoint_path = "models/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1
    )

    # train model
    alexnet.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=3,
        callbacks=[cp_callback]
    )
