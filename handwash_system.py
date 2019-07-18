#===============================================================
# Title:   handwash_system.py
# Author:  Akash Nagaraj (grassknoted@gmail.com)
# Date:    15th July, 2019
# Version: 1.0.0
#===============================================================


#===============================================================
# TODO 1: Tune the Hyperparameters, Design Parameters, and
#         Threshold Values
# TODO 2: Generalize the crop_to_region function


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

# Import the Frame Module
import frame as Frame

# Import the FrameBuffer Module
import frame_buffer as FrameBuffer

# Import the HandwashSteps Module
import handwash_steps as HandwashSteps

# Initialize a previous temporal image
previous_temporal_image = np.array([0])

class HandwashSystem:

    # Constructor
    def __init__(self):
        '''
        Constructor to initialize the Handwash system
        '''

        # Initialize the Frame Buffer of size 15
        self.frame_buffer = FrameBuffer.FrameBuffer(15)

        # Initialize the Video Stream
        self.live_stream = cv2.VideoCapture(0)
        print("Video Stream started successfully.")
        print("Changing video stream resolution..")
        live_stream.set(3, 1080)
        live_stream.set(4, 1080)
        time.sleep(1)

        print("Video Stream resolution changed to: 1080x1080")

        # Previous frame stored to generate optical flow
        self.previous_frame = null

        # Circular system counter to keep count of frames
        self.frame_count = 0

        # Sampling rate of frames for prediction
        # In a 30 FPS Video, a sampling rate of 5
        # would mean: 1 in 5 frames are sampled for prediction
        # Therefore, every second, 6 frames are sampled
        self.sampling_rate = 15

        # Property to set default FPS of the video stream
        self.video_fps = 30

        # Interval to check buffer in (milliseconds)
        self.check_buffer_interval = 1000

        self.step_name =   {"step_1"       : "STEP 1",
                            "step_2_left"  : "STEP 2 LEFT",
                            "step_2_right" : "STEP 2 RIGHT",
                            "step_3"       : "STEP 3",
                            "step_4_left"  : "STEP 4 LEFT",
                            "step_4_right" : "STEP 4 RIGHT",
                            "step_5_left"  : "STEP 5 LEFT",
                            "step_5_right" : "STEP 5 RIGHT",
                            "step_6_left"  : "STEP 6 LEFT",
                            "step_6_right" : "STEP 6 RIGHT",
                            "step_7_left"  : "STEP 7 LEFT",
                            "step_7_right" : "STEP 7 RIGHT"}



    # Function to return frames to display on the Flask Server
    # DO NOT CHANGE
    def get_frame(self):
        '''
        Returns frames from the live video stream, encoded as jpeg bytes
        jpeg bytes is the format used by the Flask server to display the 
        live stream
        '''
        # Increment frame_count
        self.frame_count += 1
        
        # Read the live feed frame
        success, image = self.live_stream.read()
        
        # Sample frames according to the sampling rate
        if self.frame_count % (int)(self.video_fps / self.sampling_rate) == 0:
            # Create a Frame Object
            frame_object = Frame.Frame(image)
            # Preprocess the frame for prediction
            frame_object.preprocess(28)

            # Generate the optical flow for the image
            frame_object.generate_optical_flow(previous_frame)

            # Predict the frame

            self.frame_buffer.add_to_buffer(frame_object)

        if( self.frame_count % 30 == 0 ):
            self.frame_buffer.get_step_predicted()

        ret, jpeg = cv2.imencode('.jpg', image)

        # Store the current Image as the previous image
        previous_image = image

        return jpeg.tobytes()



    def check_buffer(self):
    '''
    Function to check FrameBuffer, and interface it with the
    HandwashSteps module.
    This fuction is called repeatedly, with an interval of: check_buffer_interval
    '''

    step_completed = self.frame_buffer.get_step_predicted()





    # Destructor
    def __del__(self):
        '''
        Destructor to delete the HandwashSystem
        '''

        # Release the video stream
        self.live_stream.release()
        print("Live Stream successfully released.")

        # Delete the FrameBuffer
        del(self.frame_buffer)