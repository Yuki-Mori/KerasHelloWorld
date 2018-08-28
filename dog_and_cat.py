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

import glob

TARGET_SIZE = (128, 128)

def get_dataset(dirname):
    dog_files = glob.glob(dirname+'/dog/*.jpg')
    cat_files = glob.glob(dirname+'/cat/*.jpg')

    X = []
    Y = []
    
    for filename in dog_files:
        img = load_img(filename, target_size=TARGET_SIZE)
        array = img_to_array(img) / 255
        X.append(array)
        Y.append([0,1])

    for fileanme in cat_files:
        img = load_img(filename, target_size=TARGET_SIZE)
        array = img_to_array(img) / 255
        X.append(array)
        Y.append([1,0])

    return (X, Y)
