#!/usr/bin/env python

import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import datasets, layers, models, losses
from tensorflow.keras.preprocessing import image_dataset_from_directory

# set parameters
seed = 123
image_dir = pathlib.Path('Greyscale_Images/')
img_height = 2048
img_width = 2048

if __name__ == '__main__':
    # load datasets
    train_ds = image_dataset_from_directory( # 20% training
        image_dir,
        validation_split=0.2,
        subset='training',
        seed=seed,
        image_size=(img_height, img_width),
    )
    nontrain_ds = image_dataset_from_directory(
        image_dir,
        validation_split=0.2,
        subset='validation',
        seed=seed,
        image_size=(img_height, img_width),
        # batch_size=batch_size
    )
    validation_ds = nontrain_ds.random(seed=seed).take(0.5 * tf.data.experimental.cardinality(nontrain_ds)) # 10% validation
    test_ds = nontrain_ds.random(seed=seed).skip(0.5 * tf.data.experimental.cardinality(nontrain_ds)) # 10% testing

    # build AlexNet CNN
    model = models.Sequential()
    model.add(layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear", input_shape=x_train.shape[1:]))
    model.add(layers.Conv2D(96, 11, strides=4, padding='same'))
    model.add(layers.Lambda(tf.nn.local_response_normalization))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(3, strides=2))
    model.add(layers.Conv2D(256, 5, strides=4, padding='same'))
    model.add(layers.Lambda(tf.nn.local_response_normalization))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(3, strides=2))
    model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(256, 3, strides=4, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    print(alexnet.summary())
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
    history = alexnet.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=3,
        callbacks=[cp_callback]
    )

    # plot loss and accuracy
    fig, axs = plt.subplots(2, 1, figsize=(15,15))
    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].title.set_text('Training Loss vs Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend(['Train', 'Val'])
    axs[1].plot(history.history['accuracy'])
    axs[1].plot(history.history['val_accuracy'])
    axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend(['Train', 'Val'])
    plt.savefig('loss-accuracy.png', bbox_inches='tight')

    # evaluate performance on test data
    (test_loss, test_accuracy) = model.evaluate(test_ds)
    print('test loss:', test_loss)
    print('test accuracy:', test_accuracy)
