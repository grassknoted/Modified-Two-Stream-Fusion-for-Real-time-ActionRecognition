#!/bin/bash
# Setup Script for the Sample Code Submission of Paper ID: 9202

# Install the required Python Modules
pip install -r requirements.txt

# Setup PyFlow for Optical Flow Computation
cd pyflow
python setup.py build_ext -ID
cd ..

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vZm3LDdJLwBP-1LNVCV9oP8a77lRUjvp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1vZm3LDdJLwBP-1LNVCV9oP8a77lRUjvp" -O handwash_step_model.h5
rm -rf /tmp/cookies.txt