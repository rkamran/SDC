Self Driving Car
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./car_notcar/image16.png
[image2]: ./car_notcar/image0078.png
[image3]: ./output_images/HOG_Channel.png
[image4]: ./output_images/HOG_Channel_notcar.png
[image5]: ./output_images/nocar.png
[image6]: ./output_images/lonecar.png
[image7]: ./output_images/twocars.png
[image8]: ./output_images/heatmap.png
[image9]: ./output_images/heatmap2.png
[image10]: ./output_images/heatmap3.png
[video1]: ./project_video.mp4

---
### Rubric 1. Writeup / README

This document.

### Rubric 2. Histogram of Oriented Gradients

### Histogram of Oriented Gradients (HOG)

There are two main classes in my code ```CFClassifier``` and ```CFCarFinder```. The method below in the classifier applies the HOG feature detection.
 
```python
def get_hog_features(self, img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False):
        hog_image = None
            
        if vis == True:
            features, hog_image = hog(img, orientations=orient, 
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block), 
                                      transform_sqrt=True, 
                                      visualise=vis, feature_vector=feature_vec)
        else:
            features = hog(img, orientations=orient, 
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block), 
                                      transform_sqrt=True, 
                                      visualise=vis, feature_vector=feature_vec)
                    
        return features, hog_image
```

I've tried several combination of color channels and found the following parameters detecting the features better than 
  any other combination using SVC
```python
Color_Channels = "YCrCb"
Pixel_per_cell =  8
Cell_per_block =  2
Orientation = 9
```
 
Here are the results I have gotten for car and  Non car images using these classifier values.

|Car/Not car   |HOG YCrCb Channels   |
|---|---|
|![image2]|![image3]|
|![image1]|![image4]|




#### Training
```CFClassifier``` in the ```train``` method triggers the training of the model. For training I used three channels HOG features
 detection along with histogram and spatial binning. I used scikit's LinearSVC as the classifier.
 
Features are scaled using ```StandardScaler``` and a 20-80 test/train split was also applied to the labeled data. The following code block sums up the whole training pipeline.

```python
self.scaler = StandardScaler()
self.scaler.fit(X)
scaledX = self.scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(scaledX, y, 
                                                    test_size=0.2, 
                                                    random_state=np.random.randint(0, 100))   

self.svc = LinearSVC()
self.svc.fit(X_train, y_train)
score = round(self.svc.score(X_test, y_test), 4)
```
### 3. Sliding Window Search
In my code, the class ```CFCarFinder ``` has all the necessary methods to detect vehicles. The main implementation is in the
method ```findcar_with_subsampling```. The sliding window is applied to the lower half of the image and HOG features are also calculated once and
 sub-sampled for each and every window thereafter.
 
Since the training image size was 64x64, each window segment is also resized to make sure that the feature size match when predict method is alled on the classifier.

Number of windows and sliding is calculated using ```cells_per_step```. Since I am doing subsampling on a pre caclulated hog
 ```nxsteps``` and ```nysteps``` are also calculated using ```cells_per_step``` and ```pixel per cell```.

```python
# 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
window = 64
nblocks_per_window = (window // clf.pix_per_cell) - clf.cell_per_block + 1
cells_per_step = 1  # Instead of overlap, define how many cells to step
nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
nysteps = (nyblocks - nblocks_per_window) // cells_per_step
```

|No Car   |One Car   | Two Cars|
|---|---|----|
|![image5]|![image6]|![image7]|

To reduce the false positive two steps were also added (Mainly copied code from the lessons) to generate a heatmap and apply
 a threshold to only consider hot areas. Also used the scipi's label function to draw a solid rectangle.
  
Here are few examples of heat map

|   |   | |
|---|---|----|
|![image10]|![image9]|![image8]|



### Rubric 4. Video Implementation

Here's a [link to my video result](./project_out.mp4)

I've already described the filtering method above with some examples of the heatmap.
 
Also calling the drawing after 10 frames so it reduces some false positive and keeps the vehicles in focus with less flickering.

---

### Discussion

There's this problem with vehicles being detected on the other side of the freeway which I could not quite figure out to eliminate.
Sometime it detects and or takes longer to get the vehicle in focus. The other issue I am trying to resolve 
is to display two overlapping cars as two different cars as opposed to one box.
