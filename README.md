
# CERTH ImageBlurDataset Classification
by [Dhruv Patel]([Github](https://github.com/dhruvbpatel/))

**Problem Statement:** 
	Image quality detection has always been an arduous problem to solve in computer vision.In this project, We will  come up with features and models to build a classifier which will predict whether a given image is blurred.

DataSet Link: [CERTH_ImageBlurDataset](http://mklab.iti.gr/files/imageblur/CERTH_ImageBlurDataset.zip)
-

This repository contains code for Blurred Image classification.

**Dependecies:**
All the dependencies necessary for running the code is mentioned in **requirements.txt** file.

 - Firstly download the zip file and extract it

you can simple install all of them by running:

    pip install -r requirements.txt
   
   ---
**File Description**
| No. | FileName | Description  |
|--|--|--|
|1| CNN (ResNet 50) Blur_Image_Classification | Code for CNN model training and testing. Got an aggregated accuracy of ~85% using ResNet 50 Architecture.|
| 2 | Laplacian Detection on DigitalBlurSet (96% accuracy).py | Code for Laplacian Technique on Evaluation set which got 96% accuracy on DigitalBlur Set |
|3| Laplacian Detection on NaturalBlurSet (76% accuracy)| Code for Laplacian Technique on Evaluation Set which got 76% accuracy on NaturalBlur Set | 
|4|model_1_stage_3 | Saved Model after CNN training
|5| preds.csv| Prediction results of CNN model after testing on evaluation set |


All code files are available in both Jupyter notebook(.IPYNB) as well as python (.py) files.

 - This code was developed using **Google Colab GPU** runtime. So it is Highly advisible to execute this code using the same inorder to work hasslelessly.

---
**Methods Used:**

 - **CNN Based Image Classification using ResNet50 architecture**
	 -	 Using the training dataset the CNN model was trained. It incorporated the use of FASTAI library for developing the model , training and testing it.
	 -	using fastai I was able to train a CNN model using Resnet50 architecture.
	 -	The trained model was able to achieve an accuracy of ~85% on the dataset
	 -	The steps of buidling CNN involved preprocessing of Images and creating model , training it on dataset, Hyperparameter tuning , testing on validation set and then finally making prediction.
	 
 - **Laplacian Based Technique for Classifying if Images is Blurred or not**
			-	 Using Laplace transformation and calculating the variance in frequency of images we can build a classifier which can classify if an image if blurred or not.
			-	Using OpenCV and Numpy library we can vectorize the images and apply laplace method to get the variance
			-	also we can tune hyperparameter to put a threshold for classification.
			-	We were able to achieve and accuracy of ~96% on DigitalBlurSet and ~76% on NaturalBlurSet


**Feature Engineering and Inference Outcomes.**
	 -  The images in the Dataset were already categorized but the problem was that they were of different sizes
	 - Inorder to train our model effectively we reformated them into a same size and loaded into ImageDataBunch
	 - Renaming Images was also an important decision so that we can remove all mismatch in naming convention
	 - Also required preprocessing were performed before training to get most out of trained model
	 - After training upon testing in inference of trained model we found that many images were misclassified and we also displayed them
	 - Almost ~10-15% image predictions were slightly misclassified
	 - After Tuning hyperparameters and again training model we found the best hyperparameter as => 1e-03 as learning rate
	 - Further improvement of the model can be done with more images in dataset.

	