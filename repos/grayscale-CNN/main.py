import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt

from keras import layers
from keras.layers import Input, Dense, Activation,BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from matplotlib.pyplot import imshow

def AlexNet(input_shape): # from https://medium.com/analytics-vidhya/multi-class-image-classification-using-alexnet-deep-learning-network-implemented-in-keras-api-c9ae7bc4c05f

    X_input = Input(input_shape)

    X = Conv2D(96,(11,11),strides = 4,name="conv0")(X_input)
    X = BatchNormalization(axis = 3 , name = "bn0")(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3,3),strides = 2,name = 'max0')(X)

    X = Conv2D(256,(5,5),padding = 'same' , name = 'conv1')(X)
    X = BatchNormalization(axis = 3 ,name='bn1')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3,3),strides = 2,name = 'max1')(X)

    X = Conv2D(384, (3,3) , padding = 'same' , name='conv2')(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)

    X = Conv2D(384, (3,3) , padding = 'same' , name='conv3')(X)
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation('relu')(X)

    X = Conv2D(256, (3,3) , padding = 'same' , name='conv4')(X)
    X = BatchNormalization(axis = 3, name = 'bn4')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3,3),strides = 2,name = 'max2')(X)

    X = Flatten()(X)

    X = Dense(4096, activation = 'relu', name = "fc0")(X)

    X = Dense(4096, activation = 'relu', name = 'fc1')(X)

    X = Dense(6,activation='softmax',name = 'fc2')(X)

    model = Model(inputs = X_input, outputs = X, name='AlexNet')
    return model

if __name__ == '__main__':
    seed = 22
    K.set_image_data_format(‘channels_last’)

    path = '/Volumes/imaging_analysis/2021_10_06_HumanProteinAtlas_CiminiLab_TeresaGao/Greyscale_Images/' # must be connected to Broad Institute IP group VPN
    train_datagen = ImageDataGenerator(
        label_mode='categorical',
        image_size=(2048, 2048),
        rescale=1./255,
        seed=seed,
    )
    train = train_datagen.flow_from_directory(
        path,
        target_size=(2048,2048),
        class_mode='categorical',
        seed=seed,
    )

    alex = AlexNet((2048, 2048))
    alex.compile(
        optimizer = 'adam',
        loss = 'categorical_crossentropy',
        metrics=['accuracy'],
    )

    alex.fit_generator(train, epochs=50)
