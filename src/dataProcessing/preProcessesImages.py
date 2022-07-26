import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
#import PIL

def processMnistFashionData(X,y):
    X_rs = X
    y_rs = y # (60000,)
    X_rs = X_rs.reshape(X_rs.shape[0], X_rs.shape[1]*X_rs.shape[2])
    # Convert the images into 3 channels
    X_rs = np.dstack([X_rs] * 3)
    # Reshape images as per the tensor format required by tensorflow
    X_rs = X_rs.reshape(-1, 28,28,3)
    # Resize the images 48*48 as required by VGG16
    X_rs = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in X_rs])
    # Normalise the data and change data type
    X_rs = X_rs.astype('float32')
    X_rs = X_rs / 255.
    return X_rs, y_rs
    
def processCIFAR10Data(X,y):
    X_rs = X
    y_rs = y
    # Resize the images 48*48 as required by VGG16
    X_rs = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in X_rs])
    # Normalise the data and change data type
    X_rs = X_rs.astype('float32')
    X_rs = X_rs / 255.
    return X_rs, y_rs