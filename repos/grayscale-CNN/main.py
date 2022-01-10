#!/usr/bin/env python

import os
import pathlib
import math
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import islice, cycle
from tensorflow.keras import callbacks, datasets, layers, models, preprocessing, losses

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# set parameters
seed = 123
img_dir = pathlib.Path('Greyscale_Images_png/')
img_height = 2048
img_width = 2048
batch_size = 16
validation_split = 0.2
epoch_count = 25
verbose = True

random.seed(seed)
datagen = preprocessing.image.ImageDataGenerator(
    validation_split=validation_split,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20
)

def repeat_iterator(it, count):
    return islice(cycle(it), count)

def get_validation_and_test_iterators(img_count_validation, img_count_test, validation_repeats=1):
    ''' Create validation and test datasets. '''
    nontrain_ds = preprocessing.image.DirectoryIterator(
        img_dir,
        datagen,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        seed=seed,
        subset='validation'
    )
    if verbose:
        print('-------\ncreated nontrain_ds\n-------')
    validation_ds = islice(cycle(islice(nontrain_ds, img_count_validation)), validation_repeats) # 10% validation
    test_ds = islice(nontrain_ds, img_count_validation) # 10% testing
    if verbose:
        print('-------\ncreated validation_ds with size', img_count_validation, 'and test_ds with size', img_count_test, '\n-------')

    return (validation_ds, test_ds)

if __name__ == '__main__':
    # calculate number of classes and total image count
    num_classes = len(next(os.walk(img_dir))[1])
    if verbose:
        print('-------\nnum_classes is', num_classes, '\n-------')
    img_count_total = sum([len(files) for r, d, files in os.walk(img_dir)])
    if verbose:
        print('-------\nimg_count_total is', img_count_total, '\n-------')

    # build AlexNet CNN
    alexnet = models.Sequential()
    # alexnet.add(layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear", input_shape=x_train.shape[1:]))
    alexnet.add(layers.Conv2D(96, 11, strides=4, padding='same', kernel_regularizer='l2'))
    alexnet.add(layers.Lambda(tf.nn.local_response_normalization))
    alexnet.add(layers.Activation('relu'))
    alexnet.add(layers.MaxPooling2D(3, strides=2))
    alexnet.add(layers.Conv2D(256, 5, strides=4, padding='same', kernel_regularizer='l2'))
    alexnet.add(layers.Lambda(tf.nn.local_response_normalization))
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
        loss=losses.categorical_crossentropy,
        metrics=['accuracy']
    )

    test_ds = None
    history = {}
    time_as_string = str(datetime.now().timestamp())
    for epoch_num in range(epoch_count):
        # create callback to save model
        checkpoint_dir = "./"
        # if not os.path.exists(checkpoint_dir):
        #     os.makedirs(checkpoint_dir)
        #     if verbose:
        #         print('created checkpoint_dir', checkpoint_dir)
        cp_callback = callbacks.ModelCheckpoint(
            filepath=checkpoint_dir+time_as_string + '_checkpoint_epoch-' + str(epoch_num) + '_loss-{val_loss:.2f}.hdf5',
            monitor='val_loss',
            verbose=1
        )

        # load training dataset
        train_ds = preprocessing.image.DirectoryIterator(
            img_dir,
            datagen,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            seed=seed,
            subset='training'
        )
        if verbose:
            print('-------\ncreated train_ds\n-------')

        # prepare iterators
        img_count_nontrain = round(img_count_total * validation_split)
        img_count_train = img_count_total - img_count_nontrain
        img_count_validation = math.ceil(img_count_nontrain // 2)
        img_count_test = img_count_nontrain - img_count_validation
        validation_ds, test_ds = get_validation_and_test_iterators(img_count_validation, img_count_test, validation_repeats=math.ceil(img_count_train / batch_size)); # TODO

        # train model
        new_history = alexnet.fit(
            train_ds,
            validation_data=validation_ds,
            validation_freq=1,
            epochs=1,
            callbacks=[cp_callback] # TODO: add earlystopping callback to prevent epoch overfitting? https://www.geeksforgeeks.org/choose-optimal-number-of-epochs-to-train-a-neural-network-in-keras
        ).history

        # update history
        for key in new_history:
            print('key', key)
            if key not in history:
                history[key] = []
            history[key].append(new_history[key][0])

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
    plt.savefig(time_as_string + '_loss-accuracy.png', bbox_inches='tight')

    # evaluate performance on test data
    (test_loss, test_accuracy) = alexnet.evaluate(test_ds)
    print('\ntest loss:', test_loss)
    print('test accuracy:', test_accuracy)

print('\ndone :)')
