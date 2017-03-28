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
|Convolutional Layer 1|1x1 Stride with depth 6 with relu activation and avg_pooling with 2x2 strides|
|Convolutional Layer 2|1x1 Stride with depth 16 with relu activation and avg_pooling with 2x2 strides|
|Fully connected layer1|400 to 120 outputs|
|Drop out layer| 70% To prevent overfitting|
|Fully connected layer2|129 to 84|
|Output layer(logits)|finally provides the classification matrix|




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
TopKV2(values=array([[ 35.99718094,  22.21012115,  18.89967346,  13.96571636,  13.4583683 ],
       [ 47.02022171,  36.49865723,  33.2827301 ,  15.49156857,
          2.43720818],
       [ 60.55839157,  30.59966469,  14.05848503,   1.06750739,
         -1.17944551],
       [ 37.71593857,  31.65343285,  26.84954834,  19.51398849,
         18.42823029],
       [ 78.330513  ,  33.52876282,  29.87256622,  20.59123802,
         19.82063293]], dtype=float32), indices=array([[17, 14, 30,  5, 29],
       [38, 36, 40, 34, 20],
       [25, 36, 22,  1,  4],
       [28, 29, 25, 30, 24],
       [ 3,  5,  2,  1,  6]], dtype=int32))
```

#### German Traffic Sign - Analysis
I have captured these 5 interesting images from streets of Berlin using Google Street view. 

With the current model and converting it to gray
![alt text](https://github.com/rkamran/SDC/blob/master/project2/german_signs/1.png "Speed limit end Sign")
![alt text](https://github.com/rkamran/SDC/blob/master/project2/german_signs/2.png "Speed limit end Sign")
![alt text](https://github.com/rkamran/SDC/blob/master/project2/german_signs/3.png "Speed limit end Sign")
![alt text](https://github.com/rkamran/SDC/blob/master/project2/german_signs/4.png "Speed limit end Sign")
![alt text](https://github.com/rkamran/SDC/blob/master/project2/german_signs/5.png "Speed limit end Sign")


#### Where to go next?
I wasn't able to attempt the optional items and also wasn't able to improve performance beyond 92% on the training. I beleive once I am done with my other micro degree (Deep Learning Foundation) I will be able to improve the model a little bit more.
