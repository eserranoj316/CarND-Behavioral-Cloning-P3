import pandas as pd
import os
import argparse
import json
import numpy as np
import cv2
from sklearn.utils import shuffle


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.models import model_from_json

# Fix error with TF and Keras
#import tensorflow as tf
#tf.python.control_flow_ops = tf


#module for augmenting/pre-processins images
from process_image import *

def get_model():
    '''
       Implementation of NVIDIAS Neural Network for End-to-End Learning
       for Self-Driving Cars as specified in https://arxiv.org/pdf/1604.07316v1.pdf
    '''
    dropout = 0.10
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(66,200, 3)))
    model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))
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
    model.add(Dense(1164, init='he_normal', name="dense_1164", activation='elu'))
    model.add(Dense(100, init='he_normal', name="dense_100", activation='elu'))
    model.add(Dropout(dropout))
    model.add(Dense(50, init='he_normal', name="dense_50", activation='elu'))
    model.add(Dropout(dropout))
    model.add(Dense(10, init='he_normal', name="dense_10", activation='elu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, init='he_normal', name="dense_1"))
    model.compile(loss='mse', optimizer='adam')
    return model
if __name__ == "__main__":
  #Syntax 1: python model.py
  #        Initial training of the model.   
  #Syntax 2: python model.py --model model_curr.json
  #        model_curr.json specified in --model option will be used and the 
  #        corresponding  model_curr.h5 is loaded
  #Assumptions:required files data/driving_log.csv and data/IMG/*.jpg should
  #         be on the same directory where model.py is located.
  #output: model.json/model.h5            
  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--model', type=str, default="getmodel", help='')
  args = parser.parse_args()
  if os.path.exists(args.model):
    #model available for tuning  
    modelfile = args.model
    with open(modelfile, 'r') as jfile:
      print(args.model) 
      model = model_from_json(json.loads(jfile.read()))
      model.compile("adam", "mse")
      weights_file = modelfile.replace('json', 'h5')
      model.load_weights(weights_file)
  else:
      model = get_model()
      model.compile("adam", "mse")
  
  logFiles = "data/driving_log.csv"
  data = shuffle(pd.read_csv(logFiles)).reset_index()
  #split the data 80/20 data_test/data_val  
  data_test = data[0:int(len(data)*0.80)]
  data_val =  data[int(len(data)*0.80)+1:]
  #one time generation  of validation data a    
  xval,yval = generate_val(data_val, len(data_val))
  
  
  pr_threshold = 1.
  for i in range(25):
    #see process_image.py for generate_train_from_PD_batch description  
    model.fit_generator(generate_train_data(data_test,256),
                           samples_per_epoch=20000,
                           nb_epoch=1, verbose=1, validation_data=(xval,yval))
    pr_threshold = 1/(i + 1)*1
  print("Saving model weights and configuration file.")
  model.save_weights("model.h5", True)
  with open('model.json', 'w') as outfile:
      json.dump(model.to_json(), outfile)

   


        
