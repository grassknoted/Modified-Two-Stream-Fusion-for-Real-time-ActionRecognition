#===============================================================
# Title:   handwash_steps.py
# Author:  Akash Nagaraj (grassknoted@gmail.com)
# Date:    17th July, 2019
# Version: 1.0.0
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


class HandwashSteps:

    @property
    def steps(self):
        return self.placeholder_steps

    # Constructor
    def __init__(self):
        # Step Dictionary
        self.placeholder_steps =   {"STEP 1"       : "incomplete",
                                    "STEP 2 LEFT"  : "incomplete",
                                    "STEP 2 RIGHT" : "incomplete",
                                    "STEP 3"       : "incomplete",
                                    "STEP 4 LEFT"  : "incomplete",
                                    "STEP 4 RIGHT" : "incomplete",
                                    "STEP 5 LEFT"  : "incomplete",
                                    "STEP 5 RIGHT" : "incomplete",
                                    "STEP 6 LEFT"  : "incomplete",
                                    "STEP 6 RIGHT" : "incomplete",
                                    "STEP 7 LEFT"  : "incomplete",
                                    "STEP 7 RIGHT" : "incomplete" }

        self.current_step_pointer = 0



    def is_correct_step(self, step_just_completed):
        '''
        Function to check if the step just performed is the
        correct step in the sequence

        @Return values
        True:  if the step completed if correct in the order
        False: all other cases
        '''
        if "or" in self.get_next_step():
            if step_just_completed == self.get_next_step().split(" or ")[0] or step_just_completed == self.get_next_step().split(" or ")[1]:
                return True
        else:
            if step_just_completed == self.get_next_step():
                return True
        return False


    def get_step_number(self, current_step):
        '''
        Function to get the number of the current step
        '''
        return int(current_step[5])


    def incorrect_step_order(self, incorrect_step):
        '''
        Function to be called if an incorrect step is completed
        '''

        # Print the error message
        if 'or' in self.get_next_step():
            return str(incorrect_step + " is not the step, please perform either " + self.get_next_step() + '.')
        elif str(self.get_step_number(incorrect_step)) in self.get_next_step():
            return str(incorrect_step + " has been successfully registered.")
        else:
            return str(incorrect_step + " is not the step, please perform " + self.get_next_step() + '.')


    def all_steps_completed(self):
        '''
        Function to check if all the steps are completed
        '''

        for step_to_check in self.steps.keys():
            if self.steps[step_to_check] == "incomplete":
                return False
        
        return True


    def handwash_completed_successfully(self):
        '''
        Function to be called when the handwash is completed
        '''

        print("Handwash completed successfully!")
        exit(0)


    def add_step(self, step_just_completed):
        '''
        Add the step just completed to the steps dictionary
        '''
        
        # If the step is not in the correct order
        if not self.is_correct_step(step_just_completed):
            self.incorrect_step_order(step_just_completed)
            return

        # Mark the step as complete
        self.placeholder_steps[step_just_completed] = "complete"

        # CHeck if the handwash is complete
        if self.is_correct_step(step_just_completed) and self.all_steps_completed():
            self.handwash_completed_successfully()
        
        # Recompute the current_step_pointer to point to the first
        # step that is not complete
        self.current_step_pointer = 0
        for step_name in self.placeholder_steps.keys():
            if (self.placeholder_steps[step_name] == "incomplete"):
                break

            else:
                self.current_step_pointer += 1


    def get_next_step(self):
        '''
        Function to return the step the person is to perform
        Returns a string of the immediate next step to be performed
        
        The returned string is either only one step
        i.e 'STEP 2 LEFT'

        or is 2 steps separated by a semi-colon 
        i.e 'STEP 2 LEFT or STEP 2 RIGHT'
        '''

        # Left and right steps to be done, and right step is not complete
        if  ("LEFT" in list(self.steps.keys())[self.current_step_pointer] ) and    \
            ("RIGHT" in list(self.steps.keys())[self.current_step_pointer + 1] and \
              self.placeholder_steps[list(self.steps.keys())[self.current_step_pointer + 1]] == "incomplete" ):
            return str(list(self.steps.keys())[self.current_step_pointer]) + " or " + str(list(self.steps.keys())[self.current_step_pointer + 1])
        
        # Left and right steps to be done, and right step is complete
        elif ("LEFT" in list(self.steps.keys())[self.current_step_pointer]) and     \
             ("RIGHT" in list(self.steps.keys())[self.current_step_pointer + 1] and \
              self.placeholder_steps[list(self.steps.keys())[self.current_step_pointer + 1]] == "complete" ):
            return str(list(self.steps.keys())[self.current_step_pointer])

        # Left step completed, right step is to be done
        elif "RIGHT" in list(self.steps.keys())[self.current_step_pointer]:
            return str(list(self.steps.keys())[self.current_step_pointer])

        # If the step is asymmetric
        else:
            return str(list(self.steps.keys())[self.current_step_pointer])


    # Constructor
    def __del__(self):
        # Delete steps dictionary
        del(self.placeholder_steps)

        print("Steps system memory freed.")
