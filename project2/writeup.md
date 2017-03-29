# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report
---
## Writeup

### Files Submitted
https://github.com/rkamran/SDC/tree/master/project2
* writeup.md - This writeup
* Traffic_Sign_Classifier.ipynb - Main jupyter notebook with all sources
* ./real_world - Folder containing 5 images used to test the model using realword examples. Note - File Names are the actual labels so we can use them to see how good the model performed in real world
* lenet_trafficXXX - Tensor flow trained model files.

### Data Set
I used provided pickle data to craete the training, test and validation data set. Most of the code was there so it was just a matter of filling in the blanks with the right file names and variables.

### Exploratory Visualization
One cell uses the test data to randomly display images to validate that it's a 32x32 image with RGB channels and one dimensional labels.

### Designing, testing and validating the model
Most of the LeNet function is a carry forward from the MINST lab so there's not much change except that it was adjusted for color images and I also replaced max_pool with avg_pool which produced better results and I was able to get accuracy up to 91% on testing dataset.
```python
# Pooling. Input = 10x10x16. Output = 5x5x16.
c2 = tf.nn.avg_pool(c2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")
```
The remainig model is pretty much the same with two convulsion layers along with three fully connected layers to map the input images to a set of 43 labels. The first filter layer is 5x5 with depth 6 and the next layer uses a 5x5 with depth 16. The fully connected layers produces output of 120, 84 and then produces the probability matrix of size 43. One hot mapping then uses the highest probability to map the image to a sign.

Here's how the overall network design looks like

| Layer        | Description|
|:-------------|:-------------|
|Input Layer|Image 32x32 with 3 color channels|
|Convolutional Layer 1|1x1 Stride of depth 6 with relu activation and avg_pooling with 2x2 strides|
|Convolutional Layer 2|1x1 Stride of depth 16 with relu activation and avg_pooling with 2x2 strides|
|Fully connected layer1|400 to 120 outputs|
|Drop out layer| 50% To prevent overfitting|
|Fully connected layer2|129 to 84|
|Output layer(logits)|Ouput layer provides the classification matrix|




To train the model after tunning for several hyper parameters I finally setteled on these values to get an accuracy of 91%.
```python
#HYPER_PARAM
rate = 0.001
EPOCHS = 15
BATCH_SIZE = 128

# For truncated normal distribution
mu = 0
sigma = 0.1
```
### Testing the model on new images
I randomly picked five images from a google search and manually sized it to match LeNet's expected size of 32x32. Just to make things simpler, image names are representing the labels. This is only used to calculate the accuracy of the model on test images and doesn't impact network's performance at all. 

Here's how the network performed

| Images        | Prediction| Correct?  |
|:-------------:|:-------------:|:-----:|
| ![alt text](https://github.com/rkamran/SDC/blob/master/project2/real_world/14.jpg "Stop Sign")| 38 | No|
| ![alt text](https://github.com/rkamran/SDC/blob/master/project2/real_world/17.jpg "Stop Sign")| 17 | Yes|
| ![alt text](https://github.com/rkamran/SDC/blob/master/project2/real_world/25.jpg "Stop Sign")| 25 | Yes|
| ![alt text](https://github.com/rkamran/SDC/blob/master/project2/real_world/28.jpg "Stop Sign")| 28 | Yes|
| ![alt text](https://github.com/rkamran/SDC/blob/master/project2/real_world/3.jpg "Stop Sign")| 3 | Yes|

By simply comparing prediction and labels I was able to calculate the profmace of the network to be around 80% with real world data.

Here's the sample output of the probabilities of the model. 

```python
TopKV2(values=array([[  9.05268312e-01,   8.89642462e-02,   5.75298956e-03,
          1.39325812e-05,   2.10835822e-07],
       [  9.99999881e-01,   1.13391181e-07,   8.43429104e-10,
          2.40555904e-12,   7.01544942e-14],
       [  9.99997973e-01,   1.18787568e-06,   7.19388538e-07,
          1.42483429e-07,   4.01445155e-10],
       [  9.99636173e-01,   3.48971807e-04,   8.18189801e-06,
          3.94306744e-06,   1.64266510e-06],
       [  1.00000000e+00,   3.17521517e-17,   3.10509126e-19,
          3.96929643e-24,   1.01022828e-25]], dtype=float32), indices=array([[14, 29, 25, 18,  3],
       [38, 40, 18, 32, 26],
       [25, 26, 24, 29, 30],
       [28, 23, 20, 29, 41],
       [ 3,  2,  5, 42, 35]], dtype=int32))
```

#### German Traffic Sign - Analysis
I have captured these 5 interesting images from streets of Berlin using Google Street view. 
With the current model and converting it to gray

| Images        | Prediction|
|:------------- |:-------------|
|![alt text](https://github.com/rkamran/SDC/blob/master/project2/german_signs/1.png "Speed limit end Sign")|This is an interesting sign and if we do grayscaling preprocessing it might turn out to be very similar to speed sign|
|![alt text](https://github.com/rkamran/SDC/blob/master/project2/german_signs/2.png "Speed limit end Sign")|The shade and other factors on the sign makes it interesting as well. If we have multiple example, network might figure it out properly but with few examples it could be a difficult thing to match. A small filter might mistake confuse zero with a C because of the shadow|
|![alt text](https://github.com/rkamran/SDC/blob/master/project2/german_signs/3.png "Speed limit end Sign")|This is not a very common sign but the shape is non square (or circle) and could carry a lot of noise if try to fit it in a 32x32 image|
|![alt text](https://github.com/rkamran/SDC/blob/master/project2/german_signs/4.png "Speed limit end Sign")|Angle of the arrow is very important as a minor rotation will make it right or left. Good quality sample images would be very important|
|![alt text](https://github.com/rkamran/SDC/blob/master/project2/german_signs/5.png "Speed limit end Sign")|I thought this cluster was interesting because it has usual and unusual traffic signs. Written text could be a very troubling thing for this network as it can be mistaken for anything with text on it|


#### Where to go next?
I wasn't able to attempt the optional items and also wasn't able to improve performance beyond 92% on the training. I beleive once I am done with my other micro degree (Deep Learning Foundation) I will be able to improve the model a little bit more.
