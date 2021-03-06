#coding: utf8

import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten

import numpy as np
#from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

import glob
from datetime import datetime

TARGET_SIZE = (32,32)
batch_size = 128
epochs = 100
output_size = 2
last_activation = 'sigmoid'
#last_activation = 'softmax'

def get_dataset(dirname):
    dog_files = glob.glob(dirname+'/dog/*.jpg')
    cat_files = glob.glob(dirname+'/cat/*.jpg')

    X = []
    Y = []

    for filename in tqdm(dog_files):
        img = load_img(filename, target_size=TARGET_SIZE)
        array = img_to_array(img) / 255
        X.append(array)
        Y.append([0,1])

    for filename in tqdm(cat_files):
        img = load_img(filename, target_size=TARGET_SIZE)
        array = img_to_array(img) / 255
        X.append(array)
        Y.append([1,0])

    return (np.array(X).astype('float32'), np.array(Y).astype('float32'))

def create_model():
    model = Sequential()
    model.add(Conv2D(32,(3,3), input_shape=(TARGET_SIZE[0],TARGET_SIZE[1],3)))
    model.add(Activation('relu'))
    #model.add(Conv2D(32, (3,3)))
    #model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3), padding='same'))
    model.add(Activation('relu'))
    #model.add(Conv2D(64,(3,3)))
    #model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(64))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(output_size))
    model.add(Activation(last_activation))

    model.summary()

    return model

def img_show(img):
    from PIL import Image
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def main(traindir='./train', testdir='./test'):
    '''from keras.datasets import cifar10
    (x_train, y_train),(x_test,y_test) = cifar10.load_data()
    x_train=x_train.astype('float32')/255.0
    x_test=x_test.astype('float32')/255.0
    y_train=keras.utils.to_categorical(y_train,10)
    y_test=keras.utils.to_categorical(y_test,10)'''

    x_train, y_train = get_dataset(traindir)
    x_test, y_test = get_dataset(testdir)

    #x,y = get_dataset('../dataset')
    #x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.50)
    model = create_model()
    model.compile(loss='mean_squared_error',
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    history = model.fit(x_train, y_train,
                        shuffle = True,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=[es_cb])
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss: ", score[0])
    print("Test accuracy: ", score[1])

    name = datetime.now().strftime("%Y%m%d_%H%M%S.h5")
    model.save('models/{0}'.format(name))
    print("Model is saved as {0}".format(name))


if __name__ == '__main__':
    main()
