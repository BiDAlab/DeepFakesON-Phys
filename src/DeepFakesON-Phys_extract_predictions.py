
import numpy as np
import os
import cv2
from imageio import imread
from skimage.transform import resize
import tensorflow.keras as keras
from tensorflow.keras.models import Model,Sequential,load_model
import pandas as pd
import h5py
import glob
import sys
import scipy.io
import time 


def load_test_motion(carpeta):
    X_test = []
    images_names = []
    image_path = carpeta
    print(carpeta)
    print('Read test images')
    for f in [f for f in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, f))]:
        carpeta= os.path.join(image_path, f)
        print(carpeta)
        for imagen in [imagen for imagen in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, imagen))]:
            imagenes = os.path.join(carpeta, imagen)
            img = cv2.resize(cv2.imread(imagenes, cv2.IMREAD_COLOR), (36, 36))
            img = img.transpose((-1,0,1))
            X_test.append(img)
            images_names.append(imagenes)
    return X_test, images_names


def load_test_attention(carpeta):
    X_test = []
    images_names = []
    image_path = carpeta
    print('Read test images')
    for f in [f for f in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, f))]:
        carpeta= os.path.join(image_path, f)
        print(carpeta)
        for imagen in [imagen for imagen in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, imagen))]:
            imagenes = os.path.join(carpeta, imagen)
            img = cv2.resize(cv2.imread(imagenes, cv2.IMREAD_COLOR), (36, 36))
            img = img.transpose((-1,0,1))
            X_test.append(img)
            images_names.append(imagenes)
    return X_test, images_names

np.set_printoptions(threshold=np.inf)
data = []
batch_size = 128
model = load_model('..\\pretrained models\\DeepFakesON-Phys_CelebDF_V2.h5')
# print(model.summary())
# input("Press Enter to continue...")

image_path = r"D:\Pattern_Letters_HR_PAD\BBDD\3DMAD"
carpeta_deep= os.path.join(image_path, "DeepFrames")
carpeta_raw= os.path.join(image_path, "RawFrames")

test_data, images_names = load_test_motion(carpeta_deep)
test_data2, images_names = load_test_attention(carpeta_raw)

test_data = np.array(test_data, copy=False, dtype=np.float32)
test_data2 = np.array(test_data2, copy=False, dtype=np.float32)

predictions = model.predict([test_data, test_data2], batch_size=batch_size, verbose=1)
bufsize = 1
nombre_fichero_scores = 'deepfake_scores.txt'
fichero_scores = open(nombre_fichero_scores,'w',buffering=bufsize)
fichero_scores.write("img;score\n")
for i in range(predictions.shape[0]):
    fichero_scores.write("%s" % images_names[i]) #fichero
    # if float(predictions[i])<0:
        # predictions[i]='0'
    # elif float(predictions[i])>1:
        # predictions[i]='1'
    fichero_scores.write(";%s\n" % predictions[i]) #scores predichas