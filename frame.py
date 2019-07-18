#===============================================================
# Title:   frame.py
# Author:  Akash Nagaraj (grassknoted@gmail.com)
# Date:    8th May, 2019
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
global_frame_id = 1


class Frame:
    '''
    Class to contain operations with respect to individual
    Frames
    '''
    

    # The following properties are defined as follows
    # to ensure that when called, the associated Frame
    # Object will have the most up-to-date value of the
    # attribute.
    #
    # This is done by changing a placeholder variable for
    # each property, and returning the placeholder variable's
    # value when a call is made to the real variable.
    #
    # If not implemented the way it is, the attributes of the
    # associated Frame object remains the same as the attributes
    # initialised by the __init__() function

    @property
    def pixel_values_array(self):
        return np.array(self.placeholder_pixel_values_array)


    @property
    def shape(self):
        return self.pixel_values_array.shape


    @property
    def frame_height(self):
        return self.pixel_values_array.shape[0]


    @property
    def frame_width(self):
        return self.pixel_values_array.shape[1]


    @property
    def number_of_channels(self):
        if len(self.pixel_values_array.shape) == 2:
            return 1
        return self.pixel_values_array.shape[2]


    @property
    def dense_optical_flow_vector(self):
        return self.placeholder_dense_optical_flow_vector


    @property
    def class_predicted(self):
        return self.placeholder_class_predicted


    @property
    def confidence_score(self):
        return self.placeholder_confidence_score


    # Constructor 
    def __init__(self, pixel_values_array_input):
        '''
        Initialize the Frame, to its pixel values array
        
        Parameters:
        @pixel_values_array : The value returned from cv2.imread()
        '''

        global global_frame_id

        # Try-Except block to handle empty images being passed to the class
        try:
            assert not(pixel_values_array_input == None), "The frame passed was empty!"
        except ValueError:
            print("The frame is successfully passed, with FrameID: "+str(global_frame_id)+", and shape:")
        
        # Assign an ID to the frame
        self.frame_id = global_frame_id
        global_frame_id = global_frame_id + 1

        # Initialise dense_optical_flow_vector to False to avoid
        # creation of a wrong Placeholder Optical Flow Vector 
        self.placeholder_dense_optical_flow_vector = False

        # placeholder_pixel_values_array is used so the variable
        # pixel_values_array always has the updated value associated
        # with the object
        self.placeholder_pixel_values_array = pixel_values_array_input

        # Predication attributes initialised to default initial values
        self.placeholder_class_predicted = -1
        self.placeholder_confidence_score = -1

        # Print Shape attributes, when Frame Object is initialized
        print(self.frame_height,"x", self.frame_width, "x", self.number_of_channels)


    def resize_frame(self, new_image_width, new_image_height):
        '''
        Resizing an image to the new height and new width

        Parameters:
        @new_image_height : Height the image must be resized to
        @new_image_width  : Width the image must be resized to
        '''

        self.placeholder_pixel_values_array = np.array(cv2.resize(self.pixel_values_array, (new_image_height, new_image_width)))

        print("Frame successfully resized to ", new_image_height, "x", new_image_width)


    def convert_to_grayscale(self):
        '''
        Convert the frame to grayscale from color
        '''

        assert (self.number_of_channels > 1), "The image to convert_to_grayscale is already Grayscale"

        self.placeholder_pixel_values_array = cv2.cvtColor(self.pixel_values_array, cv2.COLOR_BGR2GRAY)

    # THIS WORKS WOW
    # def check_me(self, g):
    #     print(g.hello)

    def crop_to_region(self):
        '''
        Crop the frame to select only the exitRegion of Interest
        In our system, the region of interest is the middle of the screen
        '''

        assert self.frame_height != self.frame_width, "Frame is already a cropped to region of interest."
        # assert self.pixel_values_array.shape[0] != self.pixel_values_array.shape[1], "Frame is already a cropped to region of interest."

        assert self.frame_height <= self.frame_width, "Height of the frame is greater than width!"
        # assert self.pixel_values_array.shape[0] <= self.self.pixel_values_array.shape[1], "Height of the frame is greater than width!"
        
        excess_to_crop = int((self.frame_width - self.frame_height)/2)
        cropped_image = self.pixel_values_array[0:self.frame_height, excess_to_crop:(self.frame_height+excess_to_crop)]
        
        # excess_to_crop = int((self.pixel_values_array.shape[1] - self.pixel_values_array.shape[0])/2)
        # cropped_image = self.pixel_values_array[0:self.pixel_values_array.shape[0], excess_to_crop:(self.pixel_values_array.shape[0]+excess_to_crop)]
        

        self.placeholder_pixel_values_array = cropped_image
    

    def generate_optical_flow(self, previous_frame):
        '''
        Generate the Dense Optical Flow between the current frame, and the previous frame

        Parameters:
        @previous_frame : A parameter of type Frame, that contains the previous frame
        '''

        assert (self.dense_optical_flow_vector == False), "Optical Flow already generated for the current frame."
        hsv_array = np.zeros(self.pixel_values_array.shape, dtype=np.uint8)
        hsv_array[:, :, 0] = 255
        hsv_array[:, :, 1] = 255
        magnitude, angle = cv2.cartToPolar(self.pixel_values_array[..., 0], self.pixel_values_array[..., 1])
        hsv_array[..., 0] = angle * 180 / np.pi / 2
        hsv_array[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        self.placeholder_dense_optical_flow_vector = True

        self.dense_optical_flow_vector = cv2.split(hsv_array)[2]


    def preprocess(self, final_frame_side):
        '''
        Single function to process all the default pre-processing

        Parameters:
        @final_frame_side : A parameter of type Frame, that contains the previous frame
        '''

        self.convert_to_grayscale()
        self.crop_to_region()
        self.resize_frame(final_frame_side, final_frame_side)


    def show_frame(self):
        '''
        Function to show the current frame using cv2.imshow()
        '''

        print("Frame: "+str(self.frame_id)+" is being displayed.")
 
        # show the image and wait for the 'Q' key to be pressed
        cv2.imshow("Frame: "+str(self.frame_id), self.pixel_values_array)
        cv2.waitKey(0)
        cv2.destroyWindow("Frame: "+str(self.frame_id))


    def show_details(self):
        '''
        Function to display the details associated with the Frame
        '''

        # FrameID and Shape Attributes
        print("ImageID\t\t\t\t:", self.frame_id)
        print("Frame Height\t\t\t:", self.frame_height)
        print("Frame Width\t\t\t:", self.frame_width)
        print("Number of Channels\t\t:", self.number_of_channels)
        
        # Check if Class has been pridicted already;
        # if no, print No
        if self.class_predicted == -1: 
            print("Class Predicted\t\t\t: No")
        # if yes, print predicted class and confidence score
        else:                          
            print("Class Predicted\t\t\t:", self.class_predicted)
            print("Confidence Score\t\t:", self.confidence_score)
        
        # Check if Dense Optical Flow has been generated already;
        # if no, print No
        if self.dense_optical_flow_vector == False:
            print("Dense Optical Flow Generated\t: No")
        # if yes, print Yes
        else:
            print("Dense Optical Flow Generated\t: Yes")
        

    def predict_frame(self):
        '''
        Function to actually predict the frame
        '''
        # TODO: The actual prediction


    def get_frame_id(self):
        '''
        Function to show the Current Frame's FrameID
        '''

        print("The current frame's Frame_ID is:", self.frame_id)


    def frame_predictions(self, class_predicted_input, confidence_score_input):
        '''
        Function to assign predictions to the Frame object
        Use this function as the interface to change predicted
        class and confidence score

        Parameters:
        @class_predicted_input  : The class that is predicted by the model
        @confidence_score_input : The confidence score that the predicted
                                  class is correct   
        '''
        self.placeholder_class_predicted = class_predicted_input
        self.placeholder_confidence_score = confidence_score_input


    # Destructor
    def __del__(self): 
        '''
        Destructor Function to destroy, and cleanup object properties
        '''

        # TODO: Perform Cleanup here
        
        # Uncomment to add logging:
        # print("Frame with FrameID: "+str(self.frame_id)+" was destroyed successfully.")
        pass