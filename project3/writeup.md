## Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[architeture]: ./examples/cnn-architecture.png "Architecture Image"

### Files Submitted


Following files were uploaded to code repository 
[https://github.com/rkamran/SDC/tree/master/project3](https://github.com/rkamran/SDC/tree/master/project3)
* *model.py* containing the script to create and train the model
* *drive.py* for driving the car in autonomous mode. Speed value was changed to do a faster lap
* *model.h5* containing a trained convolution neural network 
* *writeup.md* The report about the project.
* *run1.mp4* A video of self driving lap on track 1 using the model

Code can be executed (provided all required python packages are installed) with the following command
 
```sh
python drive.py model.h5
```

### Model Architecture and Training Strategy
For this project I continued using [NVIDIA's self driving car End to End architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) as I learned it in the classroom and I found it to be working really well for this problem.

![alt text][architeture]

- The architecture starts with an input layer which in our case is the data captured from simulator's training mode
- Normalization and pre processing is very simple where I have normalized the color values to 0 centered 0.5 standard deviation.
- As an extra layer I also have an image cropping layer before feeding it to stack of feature maps to reduce noise.
- Normalized data is then passed through five conv layers for feature detection.
- Output data is then flatten and pass through three fully connecte layer before it gets fed into output layer to make a steering decision.

I've tried training with a 30% and 40% dropout layer between different convolutional layers but it did not make much of a difference. Keeping the epochs to a low number seems to be keeping the model in check. 

#### Model parameter tuning

For this model I continued with *Adam Optimizer*. A generator has been created which I used with a batch size of 256 to optimize memory use due to heavy image processing.

#### Training data
As the target was to drive the the vehicle without drifting away from tracks so the strategy to collect the training data was to guide vehicle towards the center of the track.

* First lap was a normal speed center driving
* Second lap was on the right side of the track with some over-steering towards center
* Third lap was on the right side of the track with some over-steering towards center
* Also captured some slow moves on sharp turns to provide more training data and decision points.

For training I've used all camera angles that is center, right and left. A correction value of ***0.3*** used for left and right camera angles to over-steer a little towards center. I believe that causes car to swirl on some places on the track.

Available data was split 75/25 into training and validation subset. As mentioned above the generator uses batch processing of image using Keras fit_genertor utility. 
 
 #### Where to go next?
 The model didn't perform well on second track, in fact it crashed right after starting the drive. My target is to collect more data from the second track and also try other networks like LeNet and ResNet to see if that makes any difference.  
