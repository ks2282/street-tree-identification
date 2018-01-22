# Identifying Street Trees in Aerial Imagery
Capstone Project, Galvanize San Francisco
Kristie Sarkar

## Motivation
Street trees make a huge difference for my enjoyment of a walk or run, and take a lot of effort from municipalities to maintain. The ultimate goal of this project is to create something that can help cities survey and maintain trees. I also hope to personally use the output to help me plan more scenic long runs and get me out the door in my running shoes more often.

## Data Preparation


## Model Specifications

The model used is based on the VGG16 convolutional neural network architecture. A few modifications were made to fit the nature of the data, allow the model to train, and prevent overfitting the model to the training data.

- 13 convolutional layers with zero padding
- Batch Normalization after each convolutional layer
- Dropout after each block of convolutional layers
- One fully connected layer with Sigmoid output functions
- Binary cross-entropy loss function

## Results

## Technology Used
- Python
- Keras with TensorFlow Backend
- OpenCV (computer vision package)
- Amazon Web Services
  - S3 for data storage
  - EC2 GPU for model training

## Next Steps

## Files in 'src' Folder

- aws_functions.py: Functions for connecting to AWS and retrieving information
- aws_image_processing.py: Scripts for retrieving data, processing images, and loading data back to S3.
- image_prep.py: ImageProcessor class for taking an image, generating subimages, and assigning a label to each subimage.
- model_training.py: Script for retrieving and preprocessing the labeled data, and TreeIDModel class for modeling the data set.
- training_data_prep.py: Script for converting subimages into arrays, splitting into test/train sets, and loading compressed numpy arrays back to S3.

## References

### Data Sources
- [SF Street Tree List](https://data.sfgov.org/City-Infrastructure/Street-Tree-List/tkzw-k3nq)
- [Imagery data from USGS](https://lta.cr.usgs.gov/high_res_ortho)

### Architecture References
- [MNIST example](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py)
- [VGG paper](https://arxiv.org/pdf/1409.1556.pdf)

### Helpful Resources:
- [Stanford lecture on training neural networks](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture6.pdf)
- [Dropouts](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
