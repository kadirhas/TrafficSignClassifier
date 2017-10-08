#**Traffic Sign Recognition** 


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

[image1]: ./examples/training.png "Training dataset"
[image2]: ./examples/validation.png "Validation dataset"
[image3]: ./examples/test.png "Test dataset"
[image4]: ./examples/grayscale80.png "Grayscale traffic sign"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
The submission includes all the files that are required. The submission includes the distribution of the sign types for each data set, in a bar graph. It also shows some examples of the signs.
The required explanations about the design and architecture will be made in the following sections.
5 german sign photos are found from the internet, and provided with the required files. The accuracy of the trained network on these images are calculated, and shown with its top 5 guesses.


---


### Data Set Summary & Exploration

I've used pandas library to get the information about the data types. 

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43


#### 1. Distribution of the dataset

![image1]

The distribution of different signs are very different, however I didn't prefer to augment the lesser data types, since the validation and test data has also similar distribution. 

![image2]

Validation data

![image3]

Test data


As an improvement suggestion to this work, the data might be augmented adding small disturbances to the images such as color alterations, noise, blur, rotation, sheer etc. For a realistic application the difference of the data numbers between different types should be equalized while augmanting the data.

### Design and Test a Model Architecture

#### 1. Preprocessing

As a first step, I decided to convert the images to grayscale because the network is not large enough to get a solid color information. Other than that, for humans most of the time shape information is enough. In the tests I've made I've seen an increase in accuracy when I've used grayscale images.

I think using histogram equalization methods such as CLAHE would increase the performance even better and that would remove the effects of lighting conditions on the image. 

Then I've normalized the images to make them have an equal distribution from -1 to 1 with zero mean. 

Here is an example of a traffic sign image after grayscaling.

![alt text][image4]


####2 . Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 28x28x6      									|
| RELU					|         									|
| Max pooling				| 2x2 stride,  outputs 5x5x16        									|
| Convolution 1x1		| 1x1 stride, outputs 	5x5x16										|
| Drop out				|												|
| Fully connected		| 400 input, 120 output 						|
| Drop out				|												|
| Fully connected		| 120 input, 100 output							|
| RELU					|												|
| Drop out				|												|
| Output 				| 100 input, 43 output
 
I've prefered to add 3 dropouts and 1 1x1 convolution layer to the LeNeT architecture because even though my training accuracy was high, the validation accuracy was low, which implies overfitting. I've also increased the fully connected layers size a little bit to increase the redundancy of the nodes.  

#### 3. Training

To train the model, I've used AdamOptimizer with learning rate 0.0005 with a batch size of 256. I've used 20 EPOCHS. With more EPOCHS the performance does not increase in a meaningful way. However, there is a room for improvement here. The number of epochs could be increased to 100 and then the epoch which provides the best result on validation set might be used.

####4. Further improvements on the results

After setting the initial architecture as LeNet (because that was already implemented for the MINST dataset, and with the modifications I could get good results around 80%), I've noticed that even though my training accuracy was high, my validation accuracy was low. First, I've modified the preprocess to further increase my training accuracy by adding grayscaling. It increased the training accuracy but the validation accuracy was still lower than the desired limits. I've tried to add dropouts to several layers, which increased the validation accuracy near 90%. I've tried to increase number of epochs, or changing learning rates but they didn't make big difference on the validation accuracy. Then I added 1x1 convolution to prevent overfitting. With this, I succeeded to get validation accuracy around 93% but since the accuracy values flactuates, I thought I need to increase that little bit more. After increasing the size of 2 fully connected layers a little bit, I've started to get consistent results above 93%. To increase it further, I've added another dropout which provided me an accuracy around 95%.

My final model results were:
* training set accuracy of 0.990
* validation set accuracy of 0.950
* test set accuracy of 0.929

During the training, the test data accuracy was calculated only once, which resulted as 0.929 on the first time. Since the network has never seen this data, we know that it does not suffer overfitting with its high accuracy.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 




