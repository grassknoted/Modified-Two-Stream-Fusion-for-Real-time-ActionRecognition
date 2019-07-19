
#===============================================================
# Title:   frame_buffer.py
# Author:  Akash Nagaraj (grassknoted@gmail.com)
# Date:    10th May, 2019
# Version: 1.0.0
#===============================================================


#===============================================================
# TODO 1: Modularize, and make it Object Oriented
# TODO 2: Tune the Hyperparameters, Design Parameters, and
#         Threshold Values
# TODO 3: Generalize the crop_to_region function
# TODO 4: Cleanup in the destructor of the Fame Class


# KNOWN ISSUES:
# 1. Threshold Values are not optimized
# 2. Issues with the order of the hands
# 3. Optimization of the sampling rate based on the CPU
#===============================================================

# OS import to read Guest Operating System files and folders
import os

# OpenCV2 import for preprocessing of the individual frames
# Tested with OpenCV v3.4.4
import cv2

# Time import to track the time taken by different components
# of the system
import time

# Default dictionary to handle buffer predictions
from collections import defaultdict

# Warnings import to warn user of certain pitfalls of the system
import warnings

# Numpy import for array manipulation
import numpy as np

# Randint import to randomly sample frames at the specified
# sampling rate
from random import randint

# Matplotlib import to plot the Model Training Accuracy
# and Training Loss
import matplotlib.pyplot as plt

# Keras imports for the Neural Netork Framework
# import keras
# from keras.optimizers import Adam
# from keras.preprocessing import image
# from keras.models import load_model, Sequential
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Scikit Learn import for the splitting of the dataset into
# training and testing sets
# from sklearn.model_selection import train_test_split

# Variable to identify individual frames


class FrameBuffer:
    '''
    Class to provide an interface to the frame_buffer
    used for detection of incorrect actions
    '''

    @property
    def frame_buffer_array(self):
        return self.placeholder_frame_buffer_array

    @property
    def prediction_sampling(self):
        return self.placeholder_prediction_sampling

    # Constructor
    def __init__(self, buffer_size_input):
        '''
        Initialize the Frame Buffer, to a specified size
        
        Parameters:
        @buffer_size : Overall size of the buffer
        '''

        # Initialize buffer_size
        self.buffer_size = buffer_size_input

        # Initialize buffer_pointer
        self.frame_buffer_pointer = 0

        # Initialize an empty Frame Buffer of a specified size
        self.placeholder_frame_buffer_array = [[-99, -99, -99]] * buffer_size_input

        # Initializing the sampling rate
        self.placeholder_prediction_sampling = 0

        # Logging successful creation of the Frame Buffer
        print("Frame Buffer of size", self.buffer_size,"created sucessfully.")


    def add_to_buffer(self, frame_object):
        '''
        Function to add frame_objects to the Frame Buffer
        The Frame Buffer itself is implemented as a circular
        Queue, where the N ('buffer_size') most recent frame
        objects are maintained in the Frame Buffer

        Parameters:
        @frame_object : The frame object to be added to the buffer
        '''

        # Add the frame_object to the position pointed to by the frame_buffer_pointer
        self.placeholder_frame_buffer_array[self.frame_buffer_pointer] = frame_object

        # Update the frame_buffer_pointer in a circular manner
        self.frame_buffer_pointer = (self.frame_buffer_pointer+1)%self.buffer_size


    def show_buffer(self):
        '''
        Function to provide an interface to show the current 
        contents of the Frame Buffer

        Each frame is displayed in the format:
        [Frame_ID, Frame_Class_Predicted, Frame_Class_Confidence]
        '''

        print("The Frame Buffer is currently:", end=" ")

        # TODO: Handle empty images

        # Iterate through the frame_objects in the buffer
        for frame_buffer_iterator in range(len(self.frame_buffer_array)):

            try:
                if(self.frame_buffer_array[frame_buffer_iterator][0] == -99):
                    print("[Empty Buffer Slot]", end="")

            except:
                print([self.frame_buffer_array[frame_buffer_iterator].frame_id, self.frame_buffer_array[frame_buffer_iterator].class_predicted, self.frame_buffer_array[frame_buffer_iterator].confidence_score], end="")
            
            # Continue printing commas unless you are printing the
            # last frame_object in the Frame Buffer
            if(frame_buffer_iterator != (len(self.frame_buffer_array)-1)):
                print(",", end=" ")

        # Print a newline in the end, to beautify the output        
        print(" ")

    def set_prediction_sampling(self, prediction_sampling_input):
        '''
        Function to set the sampling rate of the frames in buffer
        This value must be less than the size of the frame_buffer
        '''
        assert(prediction_sampling_input <= self.buffer_size), "The number of prediction sampling frames is more than the buffer size!"
        
        self.placeholder_prediction_sampling = prediction_sampling_input


    def get_step_predicted(self):
        ''' 
        Returns the step predicted the most in the buffer
        '''
        steps = defaultdict(lambda: 0)

        for prediction in self.frame_buffer_array:
            steps[prediction.class_predicted] += 1

        print("STEP PREDICTED:",max(steps, key=steps.get))
        return max(steps, key=steps.get)       

    
    def clear_buffer(self):
        '''
        Function to clear the buffer to the default values ([-99, -99])
        '''

        self.placeholder_frame_buffer_array = [[-99, -99, -99]] * self.buffer_size


    # Destructor    
    def __del__(self):
        '''
        Destructor Function to destroy, and cleanup object properties
        '''

        # TODO: Perform Cleanup here
        print("Frame Buffer of size", self.buffer_size, "successfully deleted.")

# import cv2
# import frame as F
# import frame_buffer as FB
# i = cv2.imread('LiveDemoSetup.jpeg')
# f1 = F.Frame(i)
# f2 = F.Frame(i)
# f3 = F.Frame(i)
# f1.frame_predictions('Class 1', 0.1)
# f2.frame_predictions('Class 2', 0.2)
# f3.frame_predictions('Class 3', 0.3)
# fb = FB.FrameBuffer(2)
# fb.show_buffer()
# fb.add_to_buffer(f1)
# fb.show_buffer()
# fb.add_to_buffer(f2)
# fb.show_buffer()
# fb.show_buffer()
# fb.add_to_buffer(f3)
# fb.show_buffer()