# ML_Capstone

-	Our project uses computer vision technology along with Machine Learning libraries to be able to detect objects around it. The following project is divided into two parts:

(1)	Training a particular dataset

-	The dataset we had used consisted of images of electric buses and electric cars. This set contained 1486 images of each of the two categories for training purpose, and 191 images each for testing. The dataset was taken from Kaggle. It contained images of different electric vehicles wiz. cars and buses, at different angles and backgrounds. 

-	We used various tools from many libraries for preparing the data for training. The main objective of this was to convert the images from the given format to the desired format of arrays. This was done using the ImageDataGenerator class of tensorflow keras library. As the images are in pixel format, the values are between 1 and 255, thus to scale them to range of 0-1, a rescaling factor of 1/255 was given to the generator. 

-	The data was image data, so we required 2-dimensional convolutional networks. These were for the filtering process. A couple of dense (fully connected) layers were added on top of the convolutional layers for the actual classification process. A maxpooling layer was added in between the convolutional layers and one more after convolutional layers and before dense ones. The dense layers require one dimensional input, hence to convert the 2-dimensional output of convolutional layers to 1 dimension, a flatten layer was used.

-	Finally, this selected model was trained on the dataset which was processed previously. 20 epoch runs were done through the data. 

-	The loss function was selected to be binary_crossentropy, as it is a two class classification task. Adam optimization technique was used to tune hyperparameters like learning rate.

-	Test images were imported and processed using the same object of ImageDataGenerator class. The model was evaluated on this data, and an accuracy of around 90% was obtained on the test data.

(2)	Performing Operations on a Pre-trained model

-	In this part, we have imported a pretrained model for object detection and classification, and used that to predict objects from a given image, video or a live webcam recording. For this, the model we used was ssd_mobilenet_v3. We imported it to the program using OpenCV library of python. It had 80 different classes, like car, bus, motor-cycle, person, chair, etc.

-	We set all the input values like input size, scale according to the values we were to feed to the model. All the 80 labels were imported and stored in a list. The model gives an output of the index of the class it predicted the object to belong, thus, these labels stored in a list can be later mapped with the indices to generate a text output of the prediction.

-	OpenCV library reads images in BGR form, thus we need to convert them to RGB to feed to the model. Same goes for video and webcam captures. Later, the model predicts the objects, their position boundaries and class indices. This will be used to show the object with its label in the output frame. The confidence threshold was set to 0.6, i.e., the model will predict the object only if it has a confidence of at least 60%.
-	For the user interface, TKinter library was used. Three buttons were provided to the user, for image input, video input and webcam. Once clicked on the image or video, user can navigate to the file they want to feed to the model, this was done by OS library. Then the output window with the image/video/webcam recording will be shown which will have a rectangle around the objects in it and a label displaying what object it is predicted to be.
