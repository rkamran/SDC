
## Advanced Lane Finding Project


[//]: # (Image References)

[image1]: ./output_images/undistorted10.jpg "Undistorted Chess"
[image2]: ./camera_cal/calibration2.jpg "Chessboard Image"

[image3]: ./test_images/test5.jpg  "Test lane Example"
[image4]: ./output_images/undistorted_test5.jpg "Undistor Result"

[image5]: ./output_images/perspective_dst_straight_lines2.jpg "Before Threshold "
[image6]: ./output_images/thresholded_straight_lines2.jpg "After Threshold"


[image7]: ./output_images/perspective_src_straight_lines2.jpg "Before Tranform "
[image8]: ./output_images/perspective_dst_straight_lines2.jpg "After Tranform"

[image9]: ./output_images/window_test1.jpg "Sliding Window"
[image10]: ./output_images/polyfit_test1.jpg "Polynomial Fit"

[image11]: ./output_images/lane_straight_lines1.jpg "Final lane1"
[image12]: ./output_images/lane_test1.jpg "Final lane2"
[image13]: ./output_images/lane_test3.jpg "Final lane3"


[video1]: ./project_out.mp4 "Video"

### Rubric 1 - Writeup / README

This document. The main notebook where all the source code was written was saved as ```AdvancedLaneFinder.ipynb```.

### Rubric 2 - Camera Calibration
For camera calibration I used the same routine from the lesson and new chess board images provided with the starter kit.
The steps are pretty straight forward and I used the open CV's ```findChessboardCorners``` and ```calibrateCamera``` methods.

The camera matrix and distortion co-efficient is then used for each frame of video or test image to apply open CV's ```undistort``` method. Here's an example of chessboard image with corner drawn and undistorted.

The routine is available in the ```def caliberate_camera(images, draw=False, shape=(6, 9)):``` method in the notebook 

|Distored|Undistored|
|--------|----------|
|![image2]|![image1]|



### Rubric 3 - Pipeline (single images)

#### 3.1. Provide an example of a distortion-corrected image.
A simple method was added to the notebook to undistor using camera caliberation cacluated in the 1st step.

```python
def undistort(img, mtx, dist):
    img_copy = np.copy(img)
    img_copy = cv2.undistort(img_copy, mtx, dist, None, mtx)

    return img_copy
```

Here's an example of image before and adter 

|Original Test Image|Undistorted|
|--------|----------|
|![image3]|![image4]|

#### 3.2. Color Transformation.

For color transformation I used two layers, color and gradient transformation. The steps matches pretty much what we learned in the lesson.
One thing I did differently was to use the warped image as opposed to original image. Looks like it reduces noise a lot if we do it this way.

Gradient Threshold on the S channel 

```python
    # Convert to HSV color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_threshold[0]) & (scaled_sobel <= sx_threshold[1])] = 1
```
Color Threshold on the same S Channel.

```python
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_threshold[0]) & (s_channel <= s_threshold[1])] = 1    
```

|Warped Image|After Threshold|
|--------|----------|
|![image5]|![image6]|

#### 3.4 Perspective Transform

I could not find a better solution to dynamically adjust the source and destination points so just eyeballing and using the ruler in the image viewer to fin the best hard coded points.
 

```python
# These are hard coded values. Not sure how to calculate them dynamically yet.
    # 0. Top, Right
    # 1. Bottom, Right
    # 2. Bottom, Left
    # 3. Top, Left
    source_points = np.float32(
                    [[730, 450], 
                     [1150, undistored.shape[0] - 20],
                     [210,undistored.shape[0] - 20], 
                     [590,450]])

    dst_points = np.float32(
                    [[1150, 0], 
                     [1150, undistored.shape[0]],
                     [210,undistored.shape[0]], 
                     [210,0]])
```

Given that the image shape is (720, 1280) The final list looks like this

| Source                         | Destination   | 
|--------------------------------|---------------| 
| (730, 450) -- Top, Right       | (1150, 0)     | 
| (1150, 700) -- Bottom, Right   | (1150, 720)   |
| (210, 700) -- Bottom, Left     | (0, 720)      |
| (590, 450) -- Top, Left        | (210, 0)      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

***Identifier Points*** Before transformation 

![image7]

***After Transformation*** Warped Image

![image8]

#### 3.4. Identifying lane line pixel

To find the lanes and fit a polynomial I used the sliding window technique from the lesson. The steps are pretty similar and I started form the
warped and binary threshold image. 

So the histogram method gives us the left lane base and right lane base along the x-axis which we can use to calculate the midpoint. 
 
```python
def histogram(img):
    hist = np.copy(img)
    hist = np.sum(hist[hist.shape[0]//2:,:], axis=0)
    
    midpoint = np.int(img.shape[0]/2)
    leftx_base = np.argmax(hist[:midpoint])
    rightx_base = np.argmax(hist[midpoint:]) + midpoint
    
    return (hist, midpoint, leftx_base, rightx_base)
```

Then using the left and right base points we apply sliding window process which detects the concentration of the active pixels which we use to fit a polynomial across y access.

Here is the block of code from ```find_lanes``` method from the notebook which creates a 2nd degree polynomial.

```python
# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds] 

# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```

|Window Search|Polynomial Fit|
|--------|----------|
|![image9]|![image10]|

#### 3.5. Radius of curvature of the lane and the position of the vehicle with respect to center.
The method ```radius_of_curvature``` defined in the notebook calculates and converts values to meters. 
This method uses the guidelines from the lessons and used the same assumptions/formula to convert pixels into meters.

Note: Most of the code in this method is similar and copied form the lesson 

```python
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/height # meters per pixel in y dimension
xm_per_pix = 3.7/(rightx_base-leftx_base) # meters per pixel in x dimension
```

The following lines map the poly to x,y space in the real world and provide car's position relative to center of the lane
 
```python
# Fit new polynomials to x,y in world space
left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

# Calculate the new radii of curvature
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

vehicle_mid_meter = xm_per_pix*midpoint
real_center = (xm_per_pix*width)/2
vehicle_offset = real_center - vehicle_mid_meter

``` 

#### 3.6. Plotting the lane.

The method ```plot_lane``` defined in teh notebook takes the warped image and use the following values to fill a polynomial that represents the lane space infront of vehicle.
  
```python
ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]    
```

Rest is just cosmetic and drawing details. Here are some sample output images. 

||||
|--------|----------|----|
|![image11]|![image12]|![image13]


---

### Rubric 4 - Pipeline (video)

#### 4.1. Links to the video. It was also embedded in the jupyter notebook.

Here's a [link to my video result](./project_out.mp4)

![video1]

---

### Rubric 6 - Discussion

The main issue I am facing is the optimization of the code. For now the video pipeline is processing all frames. The thing I am still not sure is how to use the radii
of curvatures and fit points to my advantage and not process every frame of the video.

The other thing which applied is to adjust the perspective first and then apply color and gradient threshold. It seems like working better for me but according to lesson I was supposed ot apply the Sobel and color threshold before perspective transform.

And finally my curvatures are fluctuating between 550 and 1050 meters. I guess it's fine given that the actual curvature was 1Km but that something I am still struggling and not very confident about.