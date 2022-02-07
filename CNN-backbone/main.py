#!/usr/bin/env python

import math
import matplotlib.pyplot as plt
import neptune.new as neptune
import os
import pathlib
import random

from itertools import islice, cycle
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from neptune.new.types import File
from tensorflow import nn, data
from tensorflow.keras import callbacks, datasets, layers, models, preprocessing, losses, utils

AUTOTUNE = data.AUTOTUNE

# initialize Neptune.ai with API token
with open('neptune-api-token.txt', 'r') as f:
    run = neptune.init(
        api_token=f.read(),
        project='ip-superurop-tgao'
    )

# set parameters
seed = 123
img_dir = pathlib.Path('Greyscale_Images_png/')
img_height = 256
img_width = 256
batch_size = 16
validation_split = 0.2
epoch_count = 100
verbose = True
num_classes = len(next(os.walk(img_dir))[1])
random.seed(seed)
# datagen = preprocessing.image.ImageDataGenerator(
#     validation_split=validation_split,
#     horizontal_flip=True,
#     vertical_flip=True,
#     rotation_range=20
# )

# log hyperparameters to Neptune.ai
parameters = {
    'seed': seed,
    'img_height': img_height,
    'img_width': img_width,
    'batch_size': batch_size,
    'validation_split': validation_split,
    'n_epochs': epoch_count,
    'num_classes': num_classes
}

if __name__ == '__main__':

    ### LOAD DATASETS ###

    train_ds = utils.image_dataset_from_directory(
        img_dir,
        validation_split=0.2,
        subset='training',
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    ).cache().prefetch(buffer_size=AUTOTUNE)

    validation_ds = utils.image_dataset_from_directory(
        img_dir,
        validation_split=0.2,
        subset='training',
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    ).cache().prefetch(buffer_size=AUTOTUNE)

    # print(train_ds.class_names)



    ### BUILD ALEXNET CNN ###

    alexnet = models.Sequential()
    alexnet.add(layers.Conv2D(96, 11, strides=4, padding='same', kernel_regularizer='l2'))
    alexnet.add(layers.Lambda(nn.local_response_normalization))
    alexnet.add(layers.Activation('relu'))
    alexnet.add(layers.MaxPooling2D(3, strides=2))
    alexnet.add(layers.Conv2D(256, 5, strides=4, padding='same', kernel_regularizer='l2'))
    alexnet.add(layers.Lambda(nn.local_response_normalization))
    alexnet.add(layers.Activation('relu'))
    alexnet.add(layers.MaxPooling2D(3, strides=2))
    alexnet.add(layers.Conv2D(384, 3, strides=4, padding='same', kernel_regularizer='l2'))
    alexnet.add(layers.Activation('relu'))
    alexnet.add(layers.Conv2D(384, 3, strides=4, padding='same', kernel_regularizer='l2'))
    alexnet.add(layers.Activation('relu'))
    alexnet.add(layers.Conv2D(256, 3, strides=4, padding='same', kernel_regularizer='l2'))
    alexnet.add(layers.Activation('relu'))
    alexnet.add(layers.Flatten())
    alexnet.add(layers.Dense(4096, activation='relu', kernel_regularizer='l2'))
    alexnet.add(layers.Dropout(0.5))
    alexnet.add(layers.Dense(4096, activation='relu', kernel_regularizer='l2'))
    alexnet.add(layers.Dropout(0.5))
    # alexnet.add(layers.Dense(4096, activation='relu', kernel_regularizer='l2'))
    # alexnet.add(layers.Dropout(0.5))
    alexnet.add(layers.Dense(num_classes, activation='softmax'))
    alexnet.compile(
        optimizer='adam',
        loss=losses.SparseCategoricalCrossentropy(), # (from_logits=True),
        metrics=['accuracy']
    )



    ### CREATE CALLBACKS TO SAVE MODEL ###

    checkpoint_dir = "./"
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)
    #     if verbose:
    #         print('created checkpoint_dir', checkpoint_dir)
    cp_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_dir+ '_checkpoint_epoch-{epoch:0>3d}_loss-{val_loss:.2f}.hdf5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    earlystopping = callbacks.EarlyStopping(
        monitor ="val_loss",
        mode ="min",
        patience = 5,
        restore_best_weights = True
    )



    ### TRAIN MODEL ###

    history = alexnet.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epoch_count,
        callbacks=[NeptuneCallback(run=run), cp_callback, earlystopping]
    ).history

    # upload model files to Neptune
    run['model/saved_model'].upload_files('*.hdf5')

    # plot loss and accuracy
    fig, axs = plt.subplots(2, 1, figsize=(15,15))
    axs[0].plot(history['loss'])
    axs[0].plot(history['val_loss'])
    axs[0].title.set_text('Training Loss vs Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend(['Train', 'Val'])
    axs[1].plot(history['accuracy'])
    axs[1].plot(history['val_accuracy'])
    axs[1].title.set_text('Training Accuracy vs. Validation Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend(['Train', 'Val'])
    plt.savefig('loss-accuracy.png', bbox_inches='tight')

    print('\ndone :)')
