#!/usr/bin/env python

import os
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from itertools import islice
from tensorflow.keras import callbacks, datasets, layers, models, preprocessing, losses

# set parameters
seed = 123
img_dir = pathlib.Path('Greyscale_Images_png/')
img_height = 256 # TODO: reset to 2048 on CHTC!
img_width = 256  # TODO: reset to 2048 on CHTC!
batch_size = 1 # TODO: reset to 32 on CHTC!
validation_split = 0.2
epoch_count = 1 # TODO: debug iterator limitation on epoch count
verbose = True

if __name__ == '__main__':
    # calculate number of classes and total image count
    num_classes = len(next(os.walk(img_dir))[1])
    if verbose:
        print('-------\nnum_classes is', num_classes, '\n-------')
    img_count_total = sum([len(files) for r, d, files in os.walk(img_dir)])
    if verbose:
        print('-------\nimg_count_total is', img_count_total, '\n-------')

    # load training dataset
    datagen = preprocessing.image.ImageDataGenerator(validation_split=validation_split)
    train_ds = preprocessing.image.DirectoryIterator(
        img_dir,
        datagen,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        # class_mode='binary',
        seed=seed,
        subset='training'
    )
    if verbose:
        print('-------\ncreated train_ds\n-------')

    # create validation and test datasets
    nontrain_ds = preprocessing.image.DirectoryIterator(
        img_dir,
        datagen,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        # class_mode='binary',
        seed=seed,
        subset='validation'
    )
    if verbose:
        print('-------\ncreated nontrain_ds\n-------')
    img_count_nontrain = round(img_count_total * validation_split)
    img_count_validation = round(img_count_nontrain * 0.5)
    img_count_test = img_count_nontrain - img_count_validation
    if verbose:
        print('-------\nimg_count_validation is', img_count_validation, '\n-------')
    validation_ds = islice(nontrain_ds, img_count_validation) # 10% validation
    test_ds = islice(nontrain_ds, img_count_validation) # 10% testing
    if verbose:
        print('-------\ncreated validation_ds with size', img_count_validation, 'and test_ds with size', img_count_test, '\n-------')

    # build AlexNet CNN
    alexnet = models.Sequential()
    # alexnet.add(layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear", input_shape=x_train.shape[1:]))
    alexnet.add(layers.Conv2D(96, 11, strides=4, padding='same'))
    alexnet.add(layers.Lambda(tf.nn.local_response_normalization))
    alexnet.add(layers.Activation('relu'))
    alexnet.add(layers.MaxPooling2D(3, strides=2))
    # alexnet.add(layers.Conv2D(256, 5, strides=4, padding='same'))
    # alexnet.add(layers.Lambda(tf.nn.local_response_normalization))
    # alexnet.add(layers.Activation('relu'))
    # alexnet.add(layers.MaxPooling2D(3, strides=2))
    alexnet.add(layers.Conv2D(384, 3, strides=4, padding='same'))
    alexnet.add(layers.Activation('relu'))
    alexnet.add(layers.Conv2D(384, 3, strides=4, padding='same'))
    # alexnet.add(layers.Activation('relu'))
    # alexnet.add(layers.Conv2D(256, 3, strides=4, padding='same'))
    alexnet.add(layers.Activation('relu'))
    alexnet.add(layers.Flatten())
    # alexnet.add(layers.Dense(4096, activation='relu'))
    # alexnet.add(layers.Dropout(0.5))
    # alexnet.add(layers.Dense(4096, activation='relu'))
    # alexnet.add(layers.Dropout(0.5))
    alexnet.add(layers.Dense(num_classes, activation='softmax'))
    alexnet.compile(
        optimizer='adam',
        loss=losses.categorical_crossentropy,
        metrics=['accuracy']
    )

    # create callback to save model
    checkpoint_dir = "./models/"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        if verbose:
            print('created checkpoint_dir', checkpoint_dir)
    cp_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_dir+'checkpoint_epoch-{epoch:02d}_loss-{val_loss:.2f}.ckpt',
        monitor='val_loss',
        verbose=1
    )

    # train model
    history = alexnet.fit(
        train_ds,
        validation_data=validation_ds,
        validation_freq=1,
        epochs=epoch_count,
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
    (test_loss, test_accuracy) = alexnet.evaluate(test_ds)
    print('test loss:', test_loss)
    print('test accuracy:', test_accuracy)

print('\ndone :)')
