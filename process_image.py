import pandas as pd

import os
import argparse
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.models import model_from_json

import pickle
import numpy as np
import math
import pandas
import csv
import cv2

# Fix error with TF and Keras
#import tensorflow as tf
#tf.python.control_flow_ops = tf
pr_threshold = 1.
new_size_col,new_size_row = 200, 66


def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def trans_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    rows,cols = image.shape[:2]
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    return image_tr,steer_ang


def preprocessImage(image):
    shape = image.shape
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(new_size_col,new_size_row),interpolation=cv2.INTER_AREA)    
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
    path_file = "data/" + path_file
    if not os.path.exists(path_file):
        return (0,0)
    image = cv2.imread(path_file)
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


def generate_train_data(data,batch_size = 32):
    batch_images = np.zeros((batch_size, new_size_row, new_size_col, 3),dtype=np.uint8)
    batch_steering = np.zeros(batch_size)
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(data))
            line_data = data.iloc[[i_line]].reset_index()
            keep_pr = 0
            while keep_pr == 0:
                x,y = preprocess_image_file_train(line_data)
                #image not available
                if type(x) == int:
                    i_line = np.random.randint(len(data))
                    line_data = data.iloc[[i_line]].reset_index()
                    continue
                pr_unif = np.random
                if abs(y)<.1:
                    pr_val = np.random.uniform()
                    if pr_val>pr_threshold:
                        keep_pr = 1
                else:
                    keep_pr = 1
            batch_images[i_batch] = x
            batch_steering[i_batch] = y
        yield batch_images, batch_steering
        
def generate_val(data,batch_size = 32):
    batch_images = np.zeros((batch_size*3, new_size_row, new_size_col, 3),dtype=np.uint8)
    batch_steering = np.zeros(batch_size*3)
    offsetangle = {"center": 0, "left": 0.25, "right":-0.25}
    i = 0
    for i_batch in range(batch_size):
        line_data = data.iloc[[i_batch]].reset_index()
        for pos in ["center", "left", "right"]:
            path_file = "data/" + line_data[pos][0].strip()
            if not os.path.exists(path_file):
                continue
            image = cv2.imread(path_file)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            y_steer = line_data['steering'][0] + offsetangle[pos]
            image,y_steer = trans_image(image,y_steer,100)
            image = augment_brightness_camera_images(image)
            image = preprocessImage(image)
            image = np.array(image)
            batch_images[i] = image
            batch_steering[i] = y_steer
            i += 1
    return batch_images, batch_steering


        
