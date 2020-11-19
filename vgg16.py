# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:45:03 2020

@author: Ayse
"""
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

model = VGG16(weights='imagenet')

image = load_img("C:\\Users\\User\\Desktop\\bonobo_tickle.jpg", target_size=(224, 224))
img = img_to_array(image)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

preds = model.predict(img)

print('Prediction:', decode_predictions(preds, top=5))