# Finding Lane Lines on the Road
[![](resources/final-video-capture.png)](https://youtu.be/KHMm9bNizo8)

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
Lines are yellow and white, some are dotted lines. Dotted lines need to be detected as a single line. Images were loaded as RGB spaces. In OpenCV `cv2.inRange` function can be used to mask images with different colours. [Colour flickers or Colour chart](https://www.rapidtables.com/web/color/RGB_Color.html) can be used to pick specific colours. Following fucntion is used filter yellow and white parts from the images.
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
    return cv2.bitwise_and(img, img,mask=mask)
```
Following are the results after applying above filter  
![](resources/yellow-white-images.png)

### Select different colour space using OpenCV
As an example, I used to show HSV colour space and used OpenCV `cv2.COLOR_RGB2HSV` colour code.

#### How HSV and HSL colour spaces are working?
* H- Hue: Hue is a degree on the color wheel from 0 to 360. 0 is red, 120 is green, 240 is blue.
* S- Saturation: Saturation is a percentage value; 0% means a shade of gray and 100% is the full color.
* L - Lightness: Lightness is also a percentage; 0% is black, 100% is white.

![](resources/colour-cylinders.png) 

Image was croped: https://en.wikipedia.org/wiki/HSL_and_HSV  

##### HSV filter
```python
def rgb_to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
```  

Following are the results after applying HSV filter  

![](resources/hsv-images.png) 

##### HSL Filter
```python
def rgb_to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
```   
Following are the results after applying HSL filter  

![](resources/hsl_filter.png)


Compare with both filters, the HSL filter is good to detect both white and yellow lane lines  

##### Select white yellow colour spaces using HSL filter

```python
def select_white_yellow(img):
    # convert RGB colour space to HSL colour space
    converted_img = rgb_to_hsl(img)
    # white colour mask
    lower_bound = np.uint8([0, 200, 0])
    upper_bound = np.uint8([255, 255, 255])
    w_mask = cv2.inRange(converted_img, lower_bound, upper_bound)
    # yellow colour mask
    lower_bound = np.uint8([10, 0, 100])
    y_mask = cv2.inRange(converted_img, lower_bound, upper_bound)

    # combine the mask
    mask = cv2.bitwise_or(w_mask, y_mask)
    return cv2.bitwise_and(img, img, mask=mask)
    
```

Following are the results after applying combine HSL filter and colour masked filter.  

![](resources/yellow-white-images.png)

## Edge detection
Upto now we have used some functions to do some image processing techniques to prepare our test images for further image processing. Now we'll extract edges using  some advanced algorithms like Canny edge detection, Hough transformation. Here we'll use OpenCV inbuilt functions to apply these algorithmns. Following steps will be used to detect edges from images.  

* Step01 - Grayscaling preared images
* Step02 - Applying Gaussian filter to smooth gray scaled images
* Step03 - Canny edge detection from smoothed images
* Step04 - Hough tranfomation to detect lines from Canny edge detected images


### Grayscaling images
To grayscale images OpenCV `cv2.cvtColor` function is used with the OpenCV `COLOR_RGB2GRAY` colour code. Following utility function will be used in our pipeline.

```python
def grayscale(img):
    """
    This function is used to convert RGB images to gray scale iamges
    :param img: input image
    :return: gray scaled image
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
```

After applying above grayscale filter for prepared images, following are the results.  

![](resources/gray-scaled-images.png)

### Smoothing grayscaled images using Gaussian Smoothing
We'll apply OpenCV `cv2.GaussianBlur` function to smooth our grayscaled images.  

```python
def grayscale(img):
    """
    This function is used to convert RGB images to gray scale iamges
    :param img: input image
    :return: gray scaled image
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
```

Following are the results, after applying **GaussianBlur**.

![](resources/gaussian-smoothing-images.png)

### Detecting edges using Canny edge detection
Now we'll apply Canny edge detection for smoothed images. Following is the utility fuction that is used to detect edges from images.

```python
def detect_edges(img, low_threshold = 50, high_threshold=150):
    """
    This utility function is used to detect edges from input images
    :param img:
    :param low_threshold:
    :param high_threshold:
    :return: A gray scaled image only with edges
    """
    return cv2.Canny(img,low_threshold, high_threshold)
    
 ```
 
Following are the results of Canny edge detector.

![](resources/canny-edges-detected-images.png)

### Hough tranfomation for line detection
We have detected edges using Canny edge detector algorithm. Now we have to select our interest areas from images to apply Hough trasformation.

#### Select region of interest
This is the most tricky part in this pipeline building. Since we are going to detect road lane left and right lines which are bounded with vehicle front, we have to select area that is fitted with our requirements. I have manually marked our final results, before we going to do it programatically :). Following are the manual marked images.

![](resources/canny-edges-detected-images-manual-masked.png)

OpenCV `cv2.fillPoly` was used to select a polygon to caputure our interest ares. Following are the functions used to select interested regions.

```python
def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
```
```python
def select_region(img):
    """
    This is used to select an interest region. Here our major interest points are left lane line and right lane line
    :param img:
    :return: Seleted region as a gray scaled edge image
    """
#     imshape = img.shape
    # Initial vertices selection with hard coded pixels
    # vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
    
    # Improved version of vertices selection with dyanamic ratios
    height,width = img.shape[:2]
    bottom_left = [width*0.1 , height*0.95]
    top_left = [width*0.4 , height * 0.6]
    bottom_right = [width * 0.9, height * 0.95]
    top_right = [width * 0.6, height * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return region_of_interest(img, vertices)
```

After applying above functions for Canny detected images, following results were extracted.

![](resources/roi-images.png)

#### Applying Hough line transformation
`cv2.HoughLinesP` was used to Hough tranfomation. More theories can be found with this [wikipedia link](https://en.wikipedia.org/wiki/Hough_transform). Following function was used to do Hough transform.

```python
def hough_lines(image):
    """
    `image` should be the output of a Canny transform.

    Returns hough lines (not the image with lines)
    """
    return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)
```

After Hough line trasformation, lines were drawan using following function.

```python
def draw_lines(img, lines, thickness=2, make_copy=True):
    # the lines returned by cv2.HoughLinesP has the shape (-1, 1, 4)
    if make_copy:
        img = np.copy(img) # don't want to modify the original
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), [255, 0, 0], thickness)
    return img
    
```

Final results with drawan lines.

![](resources/draw-lines-images-01.png)
