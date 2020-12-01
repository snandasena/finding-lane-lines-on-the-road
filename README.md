# Finding Lane Lines on the Road
![](resources/final-video-capture.png)

## Introduction
In this project we are going to detect lane line on the road. Inputs were provided as a video clips and annotations were done with provided input road images. Our goal to detect lane line using annotated road images. Following learning objectives will be covered from this projects.

* Basic usage of OpenCV Python APIs
* Basic image processing techniques such as colour scale changes, colour selection, bluring images
* Images edge detection using Canny edge detector (OpenCV will be used to apply these theories)
* Required images' region selection. Both hard coded and dynamic selections will be used
* Hough tranfomation line detection (OpenCV will be used to appy these alogorithnms)

This project consits of following major files and folders.
* Finding_Lane_Lines_on_the_Road.ipynb - **The IPython notebook file**
* Finding_Lane_Lines_on_the_Road.html - **A extracted HTML file from notebook**
* READMED.md - **Writeup for this project**
* test_images - **Test images directory**
* test_videos - **Test videos directory**
* test_videos_output - **Output directory**
* opencv - **Sample OpenCV works**
* resources - **Required images for writeup**
* requirements,txt - **Python dependencies**

## How to run this project
### Step01: Clone or download this project 
Note: Before run this project you can see final results from output directory or by opening provided HTML file in a browser.

### Step02: Running this project using Python3.6 or above version
#### Create a Python virtual enviroment
Note: All commands were tested only Ubuntu 16.04
```bash
# Install virtualenv
sudo apt install virtualenv
# Create a Python virtual machine
virtualenv -p python3.6 ~/pvm
# Activate PVM
source ~/pvm/bin/activate
# Install dependencies
python -m pip install -r requirments.txt
# Run Jupyer notebook
jupyter notebook <location to IPython notebook file>
```

