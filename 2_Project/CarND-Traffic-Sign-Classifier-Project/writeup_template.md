# **Traffic Sign Recognition**
### Liang Xu
### Aug/24/2017
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---


### Data Set Summary & Exploration
You're reading it! and here is a link to my [project code](https://github.com/abysmalocean/SelfDrivingCar_Term1/tree/master/2_Project/CarND-Traffic-Sign-Classifier-Project).
I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is [32, 32, 3]
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text](reportImage/hist.png)

Here is a list of traffic signs
![alt text](reportImage/1_15.png)
![alt text](reportImage/16_31.png)
![alt text](reportImage/32_42.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

**Answer**
I have applied the similar normalization technique with the paper [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)
First, increase the number of the image. I did it very simple way. Every picture have 20 duplicated but not the same pictures. The duplicated image created by random the effect. So each duplicated picture are created by shear, rotate, translate and blur. Not every effect is applied to the image, only randomized. All the effect have genereated by random also.

Here is an example of a traffic sign image before and after add some effect.
#### Translate the Image:
![alt text](reportImage/translate.png)

#### image shear
![alt text](reportImage/translate.png)

#### image blur
![alt text](reportImage/blur.png)

#### image gamma
![alt text](reportImage/gamma.png)

#### image rotate
![alt text](reportImage/rotate.png)

As a last step, I normalized the image data because ...
![alt text](reportImage/enhancedImage.png)

Then images were transformed in the YUV space and adjusted by histogram sketching and by increasing sharpness.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

#### Question2 Answer
My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Y channel image							|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 14x14x6 |
| RELU					|												|
| Fully connected		| Input 14x14x6 = 400 output 120        									|
| RELU					|												|
| Dropout					|												|
| Fully connected		| Input 120 output 84        									|
| RELU					|												|
| Dropout					|												|
| Fully connected		| Input 84 output 43        									|
| Softmax				| etc.        									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained the model using the same as LeNet. Use Adam optimizer, learning rate 0.001 , BATCH_SIZE = 128, EPOCHS = 200

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


For this project I have choose a well know architecture called LeNet, this perticular Deep Learning network has been shown it ability to image classification. The learning rate is 0.001 after try and error. This deep network has shown it ability to distinguish the hand writting, it show also good for traffic sign calssification.

My final model results were:
* training set accuracy of 0.966
* validation set accuracy of 0.942
* test set accuracy of 0.901


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

<img src="extra_data/1.png" width="200">
<img src="extra_data/11.png" width="200">
<img src="extra_data/14.png" width="200">
<img src="extra_data/17.png" width="200">
<img src="extra_data/35.png" width="200">

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Right-of-way at the next intersection      		| Right-of-way at the next intersection   									|
| Stop   			| Stop									|
| Ahead only					| Ahead only											|
| Speed limit	      		| Speed limit			 				|
| No entry			| No entry      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Right-of-way at the next intersection sign (probability of 0.88), and the image does contain a Right-of-way at the next intersection sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .88         			| Right-of-way at the next intersection 									|
| .11     				| Road work										|
| .02					| Beware of ice/snow											|
| .00	      			| Pedestrians					 				|
| .00				    | Children crossing      							|


For the second image, the prediction is very clear, it is a "Stop" sign.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| Stop 									|
| .00     				| Turn right ahead									|
| .00					| Turn left ahead											|
| .00	      			| Ahead only					 				|
| .00				    | Speed limit (60km/h)      							|

For the third image: The prediction also very clear it is a Ahead Only:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| Ahead only 									|
| .00     				| Turn right ahead									|
| .00					| Turn left ahead											|
| .00	      			|Go straight or left				 				|
| .00				    | Go straight or right    							|

For the fourth image: The prediction also very clear it is a speed limit (30km/h) Only. The top 5 predictions are all speed limit sign

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| Speed limit (30km/h)									|
| .00     				| Speed limit (20km/h)									|
| .00					| Speed limit (70km/h)										|
| .00	      			|Speed limit (50km/h)			 				|
| .00				    | Speed limit (100km/h) 							|

For the last image: The prediction also very clear it is a No entry. The top 5 predictions are all speed limit sign

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| No entry									|
| .00     				| Keep left									|
| .00					| Priority road									|
| .00	      			|Stop		 				|
| .00				    | Go straight or left 							|
