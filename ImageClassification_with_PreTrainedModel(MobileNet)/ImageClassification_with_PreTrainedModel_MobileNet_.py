# importing libraries
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import imagenet_utils

# loading datset
file_path = 'G:/rauf/STEPBYSTEP/Data/dogcat/1.jpeg'
my_img = image.load_img(file_path, target_size=(224,224))
plt.imshow(my_img)

# preprocessing data image
resized_img = image.img_to_array(my_img)
print(resized_img.shape)
expended_image = np.expand_dims(resized_img, axis=0)
print(expended_image.shape)
expended_image = tf.keras.applications.mobilenet.preprocess_input(expended_image)

# load pre created model
my_mobile = tf.keras.applications.mobilenet.MobileNet()

# predict the unknown image
image_predicted = my_mobile.predict(expended_image)
print(image_predicted) # here you can see bunch of array 

# to change predicted array of image to labelled data we need util library
labelled_predict = imagenet_utils.decode_predictions(image_predicted)
print(labelled_predict) # here you can see labelled prediction

# from result you can see 85% of result is showing this picture is 'Labrador_retiever', Congraluations af course tested photo is Labrador!!!