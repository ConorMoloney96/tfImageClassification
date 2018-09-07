# tfImageClassification
This project uses the Tensorflow library in Python to develop and train a model capable of classifying images from the Fashion MNIST dataset and then outputs a graph to show the user the predictions which have been made and the accuracy of those predictions in an intuitive and user-friendly manner.

This program first loads the training and testing datasets. These datasets consist of 60,000 images to be used for training the model (and the correpsonding label for each image) and 10,000 images to test the model's accuracy at classifying images. Each image is 28 pixels wide and 28 pixels high and there are 10 label values (i.e. sneaker, shirt etc.).

The program then builds the model by configuring the layers of the model then compiling it. Layers are the basic building block of a neural network model, they are used to mimic the layout of the brain. The AdamOptimizer class was then used to optimize (i.e. reduce the loss function of) the model. The loss function was determined based on the accuracy of the model in correctly identifying the images. 60,000 labelled images were then used to train the model while 10,000 images were used to test the model's effectiveness. The accuracy of the model on both the training and test datasets was measured and showed to the user. If the accuracy of the model is significantly greater on the training data we say the model is over-fitted.

In this project I used tf.keras, the Tensorflow implementation of the Keras API specification, a high level neural networks API written in Python that is capable of being run on Tensorflow or Theano. The Keras API is used to build and train models and was designed to take advantage of Tensorflows functionality: Eager execution, tf.data pipelines and Estimators.

