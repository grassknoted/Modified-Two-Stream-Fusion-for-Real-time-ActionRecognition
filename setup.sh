#!/bin/bash
# Setup Script for the Sample Code Submission of Paper ID: 9202

# Install the required Python Modules
pip install -r requirements.txt

# Setup PyFlow for Optical Flow Computation
cd pyflow
python setup.py build_ext -ID
cd ..