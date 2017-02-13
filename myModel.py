import pandas as pd

import os
import argparse
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D


import pickle
import numpy as np
import math
import pandas
import csv
import cv2

# Fix error with TF and Keras
import tensorflow as tf
from tensorflow.models.image.mnist.convolutional import BATCH_SIZE
tf.python.control_flow_ops = tf



from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


new_size_col,new_size_row = 200, 66
pr_threshold = 1.
logFiles = "data/driving_log.csv"


def augment_brightness_camera_images(image):

    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


def trans_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    rows,cols = image.shape[:2]
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    return image_tr,steer_ang

def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image


def preprocessImage(image):
    shape = image.shape
    # note: numpy arrays are (row, col)!
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(new_size_col,new_size_row),interpolation=cv2.INTER_AREA)    
    #image = image/255.-.5
    return image


def preprocess_image_file_train(line_data):
    i_lrc = np.random.randint(3)
    if (i_lrc == 0):
        path_file = line_data['left'][0].strip()
        shift_ang = .25
    if (i_lrc == 1):
        path_file = line_data['center'][0].strip()
        shift_ang = 0.
    if (i_lrc == 2):
        path_file = line_data['right'][0].strip()
        shift_ang = -.25
    y_steer = line_data['steering'][0] + shift_ang
    
    #ems do path_file
    image = cv2.imread("data/" + path_file)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image,y_steer = trans_image(image,y_steer,100)
    image = augment_brightness_camera_images(image)
    image = preprocessImage(image)
    image = np.array(image)
    ind_flip = np.random.randint(2)
    if ind_flip==0:
        image = cv2.flip(image,1)
        y_steer = -y_steer
    return image,y_steer




def generate_train_from_PD_batch(data,batch_size = 32):
    
    batch_images = np.zeros((batch_size, new_size_row, new_size_col, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(data))
            line_data = data.iloc[[i_line]].reset_index()
            
            keep_pr = 0
            #x,y = preprocess_image_file_train(line_data)
            while keep_pr == 0:
                x,y = preprocess_image_file_train(line_data)
                pr_unif = np.random
                if abs(y)<.1:
                    pr_val = np.random.uniform()
                    if pr_val>pr_threshold:
                        keep_pr = 1
                else:
                    keep_pr = 1
            
            #x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
            #y = np.array([[y]])
            batch_images[i_batch] = x
            batch_steering[i_batch] = y
        yield batch_images, batch_steering
        


        
def get_model1(time_len=1):
  ch, row, col = 3, 64, 64  # camera format

  model = Sequential()
  model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch)))
  model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(Flatten())
  model.add(Dropout(.2))
  model.add(ELU())
  model.add(Dense(512))
  model.add(Dropout(.5))
  model.add(ELU())
  model.add(Dense(1))

  model.compile(optimizer="adam", loss="mse")

  return model


def get_model(time_len=1):
    dropout = 0.10
    model = Sequential()
    # Vivek, color space conversion layer so the model automatically figures out the best color space
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(66,200, 3)))
    model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))
    # Subsample == stride
    # keras.layers.convolutional.Convolution2D(nb_filter, nb_row, nb_col, border_mode='valid')
    model.add(Convolution2D(24, 5, 5, init='he_normal', activation='elu',
                            subsample=(2, 2), name='conv1'))
    model.add(Dropout(dropout))
    model.add(Convolution2D(36, 5, 5, init='he_normal', activation='elu',
                            subsample=(2, 2), name='conv2'))
    model.add(Dropout(dropout))
    model.add(Convolution2D(48, 5, 5, init='he_normal', activation='elu',
                            subsample=(2, 2), name='conv3'))
    model.add(Dropout(dropout))
    model.add(Convolution2D(64, 3, 3, init='he_normal', activation='elu',
                            subsample=(1, 1), name='conv4'))
    model.add(Dropout(dropout))
    model.add(Convolution2D(64, 3, 3, init='he_normal', activation='elu',
                            subsample=(1, 1), name='conv5'))
    model.add(Dropout(dropout))
    model.add(Flatten())
    # We think NVIDIA has an error and actually meant the flatten == 1152, so no Dense 1164 layer
    model.add(Dense(1164, init='he_normal', name="dense_1164", activation='elu'))
    model.add(Dense(100, init='he_normal', name="dense_100", activation='elu'))
    #model.add(Dropout(dropout))
    model.add(Dense(50, init='he_normal', name="dense_50", activation='elu'))
    #model.add(Dropout(dropout))
    model.add(Dense(10, init='he_normal', name="dense_10", activation='elu'))
    #model.add(Dropout(dropout))
    model.add(Dense(1, init='he_normal', name="dense_1"))
    model.compile(loss='mse', optimizer='adam')
    return model
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port', type=int, default=5557, help='Port of server.')
  parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
  parser.add_argument('--batch', type=int, default=64, help='Batch size.')
  parser.add_argument('--epoch', type=int, default=10, help='Number of epochs.')
  parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
  parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')
  parser.set_defaults(skipvalidate=False)
  parser.set_defaults(loadweights=False)
  args = parser.parse_args()

  model = get_model()
  
  logFiles = "data/driving_log.csv"

  data = pd.read_csv(logFiles)
  pr_threshold = 1
  for i in range(15):
  
      #X_train, X_validate, y_train, y_validate = getdata()
      #model.fit_generator(generate_train_from_PD_batch(data,256),
      #                      samples_per_epoch=1000,
      #                      nb_epoch=1, verbose=1, validation_data=generate_train_from_PD_batch(data,256),nb_val_samples=200)
      model.fit_generator(generate_train_from_PD_batch(data,256),
                            samples_per_epoch=20224,
                            nb_epoch=1, verbose=1)
      pr_threshold = 1/(i + 1)*1
      
  print("Saving model weights and configuration file.")

  #if not os.path.exists("./outputs/steering_model/nvidia"):
  #    os.makedirs("./outputs/steering_model/nvidia")

  model.save_weights("model.h5", True)
  with open('model.json', 'w') as outfile:
      json.dump(model.to_json(), outfile)   


        