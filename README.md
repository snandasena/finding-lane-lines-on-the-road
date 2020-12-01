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

[Running Instruction](HOW_TO_RUN.md)

## Loading Test Images
**matplotlib** was used to visualize provided and generated test image. Following are the provided test iamges.
![](resources/test-images.png)

## Image Processing
### Colour selection
Lines are yellow and white, some are dotted lines. Dotted lines need to be detected as a single line. Images were loaded as RGB spaces. In OpenCV `cv2.inRange` function can be used to mask images with different colours. [Colour flickers or Colour chart](https://www.rapidtables.com/web/color/RGB_Color.html) can be used to fick specific colours. Following fucntion is used filter yellow and white parts from the iamges.
```python
def select_rgb_white_yellow(img):
    # white mask
    lower_bound = np.uint8([220,220,220])
    upper_bound = np.uint8([255,255,255])
    w_mask = cv2.inRange(img, lower_bound, upper_bound)

    # yellow masked
    lower_bound = np.uint8([190, 190,0])
    y_mask = cv2.inRange(img, lower_bound, upper_bound)

    # combine the masks
    mask = cv2.bitwise_or(w_mask, y_mask)
    return cv2.bitwise_and(img, img,mask=mask)```
