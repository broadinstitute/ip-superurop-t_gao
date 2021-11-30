#!/usr/bin/env python

import os
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from itertools import islice
# from tensorflow.keras import callbacks, datasets, layers, models, preprocessing, losses

# set parameters
seed = 123
img_dir = pathlib.Path('Greyscale_Images_png/')
img_height = 256 # TODO: reset to 2048 on CHTC!
img_width = 256  # TODO: reset to 2048 on CHTC!
# batch_size = 32
validation_split = 0.2
verbose = True

def process(image,label):
    image = tf.cast(image/255., tf.float32)
    return image,label

if __name__ == '__main__':

    img_count_total = sum([len(files) for r, d, files in os.walk(img_dir)])
    if verbose:
        print('-------\nimg_count_total is', img_count_total, '\n-------')

    # load datasets

    datagen = keras.preprocessing.image.ImageDataGenerator(validation_split=validation_split)
# tf.keras.preprocessing.image.DirectoryIterator(
#     directory, image_data_generator, target_size=(256, 256), color_mode='rgb',
#     classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None,
#     data_format=None, save_to_dir=None, save_prefix='', save_format='png',
#     follow_links=False, subset=None, interpolation='nearest', dtype=None
# )

    train_ds = keras.preprocessing.image.DirectoryIterator(
        img_dir,
        datagen,
        target_size=(img_height, img_width),
        # batch_size=batch_size,
        # class_mode='binary',
        seed=seed,
        subset='training'
    )
    if verbose:
        print('-------\ncreated train_ds\n-------')
    nontrain_ds = keras.preprocessing.image.DirectoryIterator(
        img_dir,
        datagen,
        target_size=(img_height, img_width),
        # batch_size=batch_size,
        # class_mode='binary',
        seed=seed,
        subset='validation'
    )
    if verbose:
        print('-------\ncreated nontrain_ds\n-------')

    # nontrain_ds = tf.random.shuffle(nontrain_ds, seed=seed)
    # if verbose:
    #     print('-------\nshuffled nontrain_ds\n-------')
    # train_ds = tf.random.shuffle(train_ds, seed=seed)
    # if verbose:
    #     print('-------\nshuffled train_ds\n-------')

    #######

    # train_ds = keras.preprocessing.image_dataset_from_directory( # 20% training
    #     img_dir,
    #     validation_split=validation_split,
    #     subset='training',
    #     seed=seed,
    #     image_size=(img_height, img_width),
    #     # batch_size=batch_size
    # )
    # nontrain_ds = keras.preprocessing.image_dataset_from_directory(
    #     img_dir,
    #     validation_split=validation_split,
    #     subset='validation',
    #     seed=seed,
    #     image_size=(img_height, img_width),
    #     # batch_size=batch_size
    # )

    # print('-------\ncardinality')
    # print(tf.data.experimental.cardinality(nontrain_ds))
    # print('-------')

    # print(nontrain_ds.next())
    # print(nontrain_ds.next())

    img_count_validation = round(img_count_total * validation_split * 0.5)
    if verbose:
        print('-------\nimg_count_validation is', img_count_validation, '\n-------')

    validation_ds = islice(nontrain_ds, img_count_validation) # 10% validation
    test_ds = islice(nontrain_ds, img_count_total - img_count_validation) # 10% testing

    if verbose:
        print('-------\ncreated validation_ds and test_ds\n-------')

    # # build AlexNet CNN
    # model = models.Sequential()
    # model.add(layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear", input_shape=x_train.shape[1:]))
    # model.add(layers.Conv2D(96, 11, strides=4, padding='same'))
    # model.add(layers.Lambda(tf.nn.local_response_normalization))
    # model.add(layers.Activation('relu'))
    # model.add(layers.MaxPooling2D(3, strides=2))
    # model.add(layers.Conv2D(256, 5, strides=4, padding='same'))
    # model.add(layers.Lambda(tf.nn.local_response_normalization))
    # model.add(layers.Activation('relu'))
    # model.add(layers.MaxPooling2D(3, strides=2))
    # model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
    # model.add(layers.Activation('relu'))
    # model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
    # model.add(layers.Activation('relu'))
    # model.add(layers.Conv2D(256, 3, strides=4, padding='same'))
    # model.add(layers.Activation('relu'))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(4096, activation='relu'))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(4096, activation='relu'))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(10, activation='softmax'))
    # print(alexnet.summary())
    # alexnet.compile(
    #     optimizer='adam',
    #     loss=losses.sparse_categorical_crossentropy,
    #     metrics=['accuracy']
    # )

#     # create callback to save model
#     checkpoint_path = "models/cp.ckpt"
#     checkpoint_dir = os.path.dirname(checkpoint_path)
#     if not os.path.exists(checkpoint_dir):
#         os.makedirs(checkpoint_dir)
#     cp_callback = callbacks.ModelCheckpoint(
#         filepath=checkpoint_path,
#         verbose=1
#     )

#     # train model
#     history = alexnet.fit(
#         train_ds,
#         validation_data=validation_ds,
#         epochs=3,
#         callbacks=[cp_callback]
#     )

#     # plot loss and accuracy
#     fig, axs = plt.subplots(2, 1, figsize=(15,15))
#     axs[0].plot(history.history['loss'])
#     axs[0].plot(history.history['val_loss'])
#     axs[0].title.set_text('Training Loss vs Validation Loss')
#     axs[0].set_xlabel('Epochs')
#     axs[0].set_ylabel('Loss')
#     axs[0].legend(['Train', 'Val'])
#     axs[1].plot(history.history['accuracy'])
#     axs[1].plot(history.history['val_accuracy'])
#     axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
#     axs[1].set_xlabel('Epochs')
#     axs[1].set_ylabel('Accuracy')
#     axs[1].legend(['Train', 'Val'])
#     plt.savefig('loss-accuracy.png', bbox_inches='tight')

#     # evaluate performance on test data
#     (test_loss, test_accuracy) = model.evaluate(test_ds)
#     print('test loss:', test_loss)
#     print('test accuracy:', test_accuracy)

print('\ndone :)')
