#! /bin/bash
#
# install_python.sh
#
# About: Install Python3.5 on Ubuntu 16.04
#

sudo add-apt-repository ppa:deadsnakes/ppa

sudo apt-get update

sudo apt-get install python3.5

PIP_VERSION="9.0.3"

sudo apt-get install -y python3-pip
sudo -H pip3 install --upgrade pip=="$PIP_VERSION"