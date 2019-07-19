import os

import cv2

import time

import keras

import numpy as np

import pandas as pd

from random import randint

import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from sklearn.model_selection import train_test_split


def process_flow(im1, flow_vector):
    '''
    Function to convert flow vector to image
    '''
    hsv = np.zeros(im1.shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow_vector[..., 0], flow_vector[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    h, s, grayscale = cv2.split(hsv)

    return(grayscale)

'''
Setting Global Variables
'''
frames_per_second = 10
video_fps = 30
predicted_class_text = ""
class_prob = ""
    
# test_video_path = "/home/akash/Desktop/47.mp4"
previous_temporal_image = np.array([0])


'''
Flow Options
'''
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))


'''
Loading Model
'''
# model = load_model("/media/mukund/OS_Install/Mechanical Engineering/Sem VIII/Hand Sanitisation/Akash/trail.h5")
# fusion_model = load_model("/media/mukund/OS_Install/Mechanical Engineering/Sem VIII/Hand Sanitisation/Live Demo/spatial_model_19_0.94.h5")
fusion_model = load_model("final_combined_fused_model_pyflow_demo_10_1.00.h5")

''' 
Setting Image Parameters
'''
image_height = 112
image_width = 112

def crop_to_height(image_array):
    '''
    Crop the stream to select only the Region of Interest
    In our system, the region of interest is the middle of the screen
    '''
    height, width, channels = image_array.shape

    if height == width:
        return image_array

    image_array = np.array(image_array)

    assert height < width, "Height of the image is greater than width!"
    excess_to_crop = int((width - height)/2)
    cropped_image = image_array[0:height, excess_to_crop:(height+excess_to_crop)]
    return cropped_image

'''
Opening live video feed
'''
# vidcap = cv2.VideoCapture(test_video_path)
vidcap = cv2.VideoCapture(0)

print("Changing camera resolution..")
vidcap.set(3, 1080)
vidcap.set(4, 1080)
time.sleep(1)

print("Camera resolution changed.")

success,image = vidcap.read()
image = crop_to_height(image)

count = 0

total_frames = 0
predicted_frames = 0

# Setting display parameters for live-demo
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (50,50)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 3


'''
Assigning Classes correctly
'''
class_conversion = {0:'step_6_right', 1:'step_7_right', 2:'step_7_left', 3:'step_4_right', 4:'step_2_left', 5:'step_6_left', 6:'step_5_left', 7:'step_3', 8:'step_2_right', 9:'step_5_right', 10:'step_4_left', 11:'step_1'}
printed_class_conversion = {0:'Step 6 Right', 1:'Step 7 Right', 2:'Step 7 Left', 3:'Step 4 Right', 4:'Step 2 Left', 5:'Step 6 Left', 6:'Step 5 Left', 7:'Step 3', 8:'Step 2 Right', 9:'Step 5 Right', 10:'Step 4 Left', 11:'Step 1'}

frame_buffer = []
steps_completed = [0, 0, 0, 0, 0, 0, 0]

step_2r = False
step_2l = False
step_4r = False
step_4l = False
step_4r = False
step_5l = False
step_6r = False
step_6l = False
step_7r = False
step_7l = False

buffer_pointer = 0

max_length_of_frame_buffer = 15

for i in range(max_length_of_frame_buffer):
    frame_buffer.append(0)

def add_to_buffer(class_predicted, score):
    global frame_buffer
    global buffer_pointer
    frame_buffer[buffer_pointer] = list(class_predicted, score)
    buffer_pointer = (buffer_pointer + 1)%max_length_of_frame_buffer

def steps_completed(step_number):
    if(step_number == 2):
        step_index = step_number - 1
        if(step_2l == True and step_2r == True):
            step_checker = 0
            while(step_checker < step_index):
                if(steps_completed[step_checker] !=  )

'''
Prediction as part of live demo
'''
while success:

    image = crop_to_height(image)

    current_predictions = []

    '''
    Deciding Predition Text
    '''
    cv2.putText(image, predicted_class_text,
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    '''
    Deciding Probability Score
    '''
    cv2.putText(image,"Score: " + str(class_prob), 
        (50, 90), 
        font, 
        fontScale,
        fontColor,
        lineType)
    
    # Showing the image
    cv2.imshow('Frame',image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

        
    '''
    Generating Spatial Image
    '''
    image = cv2.resize(image, (image_height, image_width))
    spatial_image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    spatial_image_after_reshape = np.reshape(spatial_image_grayscale, (1, image_height,image_width,1))

    '''
    Generating Temporal Image
    '''  

    if(total_frames%16 == 0):
        if(len(previous_temporal_image) > 10):

            next_temporal_image = image

            previous_temporal_image = previous_temporal_image.astype(float) / 255.
            next_temporal_image = next_temporal_image.astype(float) / 255.

            # Generating Optical Flow
            u, v, im2W = pyflow.coarse2fine_flow(previous_temporal_image, next_temporal_image, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)
            temporal_image = np.concatenate((u[..., None], v[..., None]), axis=2)
            temporal_image = process_flow(next_temporal_image, temporal_image)

            
            # temporal_image_grayscale = cv2.cvtColor(temporal_image, cv2.COLOR_BGR2GRAY)
            temporal_image_after_reshape = np.reshape(temporal_image, (1, image_width, image_height, 1))

            # cv2.imshow("Temporal Image", temporal_image)
            # cv2.imshow("Normal Image", image)
            # cv2.imshow("First Image", previous_temporal_image)
            # cv2.imshow("Second Image", next_temporal_image)

            # cv2.waitKey(0)
        
            current_prediction = fusion_model.predict([np.array(spatial_image_after_reshape), np.array(temporal_image_after_reshape)])
            current_predictions.append(current_prediction)
            predicted_frames += 1

            class_prediction = np.argmax(current_prediction)
            class_prob = round(current_prediction[0][class_prediction], 4)

            predicted_class_text = printed_class_conversion[class_prediction]

            print("Predictions: ", class_conversion[class_prediction])
    previous_temporal_image = image            
    success, image = vidcap.read()
    count += 1
    total_frames += 1

print("Frames predicted:",str(predicted_frames)+'/'+str(total_frames))