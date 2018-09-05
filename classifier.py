import tensorflow as tf;
from tensorflow import keras;

import numpy as np;
import matplotlib.pyplot as plt;

fashion_mnist = keras.datasets.fashion_mnist;

#images are 28 by 28 numpy arrays. Labels are an array of integers ranging from 0-9. These correspond to the class of clothing
#Using 60,000 images to train the network and 10,000 to test it
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data();

#Each image is mapped/corresponds to a single label

#Each train_label and test_label value is an int in the range 0-9 which represents one of the 10 class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'];

#60,000 images in the training set and 60,000 labels. Each label labels an image (i.e. says what it is)	
#The shape of train_images shows that there are 60,000 images and each image consists of 28 by 28 pixels (i.e. 28 arrays each containing 28 values)		   
print(train_images.shape);
print(train_labels.shape);   

#Each pixel has a value between 0 and 255. Convert these values to 0 or 1
train_images = train_images/255.0;
test_images = test_images/255.0;

#Build the model by configuring the layers of the model then compiling it
#The basic building block of a neural network is a layer
#tf.keras.layers.Dense is a layer with parameters learned during training

model = keras.Sequential([
     #Flattens the input 
	 #the input, 28 by 28 pixels of the image, are converted to a a single layer/array
     keras.layers.Flatten(input_shape=(28,28)),
	 #2 Densely connected layers i.e. every node is connected to every other node in the preceding layer
	 keras.layers.Dense(128, activation=tf.nn.relu),
	 #Last layer is a softmax layer where values in the 10 nodes sum to 1
	 #The value in each node represents the probability that the current image belongs to the corresponding class
	 keras.layers.Dense(10, activation=tf.nn.softmax )
])


#The loss function measures how accurate the model is during training. We want to mimimize the loss function to steer the model in the right direction
#Optimizer: This is how the model is updated based on the data it sees and the loss function
#Metric are used to monitor the training and testing steps. The following example uses accuracy (i.e. the fraction of images which are correctly classified) as the metric

model.compile(optimizer = tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
			  metrics=['accuracy']
             )

#The model is fit to the data to begin training
#An epoch is the processing by the learning algorithm of the entire training set once over
#In this case the model is trained by processing the data 3 times
model.fit(train_images, train_labels, epochs=3);

train_loss, train_accuracy = model.evaluate(train_images, train_labels);
print('The accuracy of the model when applied to the training data is ', train_accuracy);

#Compare how the model performs on the test 
test_loss, test_accuracy = model.evaluate(test_images, test_labels);
print('The accuracy of the model when applied to the test data is ', test_accuracy);

#If the model is more accurate when applied to the training data than to other data (i.e. the test data) then we say this is a case of over-fitting

#With the model trained we can use it to make predictions about images

predictions = model.predict(test_images);


def graphImage(predictConfidence, test_image):
   plt.grid(False);
   plt.imshow(test_image);
   
def plot_image(i, predictions, test_labels, test_images):
   plt.grid(False)
   plt.xticks([])
   plt.yticks([])
   plt.imshow(test_images[i])
   plt.xlabel('{}({:2.0f}%) {}' .format(class_names[np.argmax(predictions[i])], np.max(predictions[i])*100, class_names[test_labels[i]] ));
   
   
def plot_value_array(i, predictions, test_labels):
   predictions_array, actual_label = predictions[i],test_labels[i];
   plt.grid(False);
   plt.xticks([]);
   plt.yticks([]);
   histogramBars = plt.bar(range(10), predictions_array, color="#777777");
   plt.ylim([0,1]); #sets the limit on the y axis
   expectedValue = np.argmax(predictions_array);
   histogramBars[expectedValue].set_color('red');
   histogramBars[actual_label].set_color('blue');
   
   
   
num_rows = 4;
num_cols = 4;
num_spaces = num_rows*num_cols;
plt.figure(figsize=(2*2*num_cols,2*num_rows));

plt.suptitle("Image recognition model test");

for i in range(num_spaces):
  #creates a subplot with the given width and height
  plt.subplot(num_rows, 2*num_cols,2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
  
   
   
plt.show();   
   
   
