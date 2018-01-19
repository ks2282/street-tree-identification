# Identifying Street Trees in Aerial Imagery
Capstone Project, Galvanize San Francisco
Kristie Sarkar

## Background

## Data Preparation

## Model Specifications

## Results

## Next Steps

## Files in 'src' Folder

1. aws_functions.py: Functions for connecting to AWS and retrieving information
2. aws_image_processing.py: Scripts for retrieving data, processing images, and loading data back to S3.
3. image_prep.py: ImageProcessor class for taking an image, generating subimages, and assigning a label to each subimage.
4. model_training.py: Script for retrieving and preprocessing the labeled data, and TreeIDModel class for modeling the data set.
5. training_data_prep.py: Script for converting subimages into arrays, splitting into test/train sets, and loading compressed numpy arrays back to S3.

## References

### Data Sources
1. SF Street Tree List: https://data.sfgov.org/City-Infrastructure/Street-Tree-List/tkzw-k3nq
2. Imagery data from USGS: https://lta.cr.usgs.gov/high_res_ortho

### Architecture References
1. MNIST example: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
2. VGG paper: https://arxiv.org/pdf/1409.1556.pdf

### Helpful Resources:
1. Stanford lecture on training neural networks: http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture6.pdf\
