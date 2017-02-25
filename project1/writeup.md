#**Finding Lane Lines on the Road** 
---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

Using the provided helper functions I just created a sequence of calls that takes an image and apply the following to find and draw lane lines
####1. Convert to grayscale
####2. Apply Guassian Blur (My Kernel Size is 3)
####3. Apply Canny Image detection with approrpate values
####4. Create a region masked and apply Hough Line detection using that region.
####5. Draw the lines.

The modifications I made to draw line function are very simple and trivial. Just calcualted the average line coordinates using unique slope value and then found two edge points one at the botton and the other using vertices defined earlier. With a little adjustment in thickness of the lines the results seems pretty close.


###2. Identify potential shortcomings with your current pipeline

The canny edge values or pipeline doesn't seems to be working very well in the video when it comes to broken pavement markers. At times there are frames with missing lines and it shows up as a flicker in the video. 



###3. Suggest possible improvements to your pipeline

Need to adjust the values I guess to make edge detection a little better. 
