# Handwash-System
Real-time Automated Handwash Auditing System


## Instructions to run the code:

#### 1. Install the requirements:
a. In the parent directory, run:
    
      $ pip install -r requirements.txt
      
      
      
#### 2. Compile and setup PyFlow:
a. Change directory to `pyflow`

      $ cd pyflow
      
b. Setup `pyflow`:

      $ python setup.py build_ext -i
    
    
    
#### 3. Dowload the trained model
a. Download the model from [here](https://drive.google.com/file/d/1vZm3LDdJLwBP-1LNVCV9oP8a77lRUjvp/view?usp=sharing)

b. Save the model in the parent folder of this repository, with the name: `current_final_handwash_model.h5`



#### 4. Run the Flask app:

      $ python main_backup.py
      
      
      
#### 5. Open `localhost:8001` on your browser when the `* Debugger pin is active` message appears on the Terminal.



## Points to remember:
1. A webcam must be connected to the computer/laptop.
2. The model must be saved in this repository's parent folder.
3. The computer must have internet access, since JavaScript is dynamically linked.
