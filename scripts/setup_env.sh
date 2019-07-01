#! /bin/bash

# echo "Installing Python3..."
# sh ./install_python.sh
echo "Installing OpenCV..."
apt-get update
apt-get -y install libgtk2.0-dev
python3 -m pip install opencv-python

echo "Installing Tensorflow..."
conda install tensorflow -c anaconda
