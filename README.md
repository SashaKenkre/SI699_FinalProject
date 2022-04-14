# SI 699 Final Project: Mask Detection
### Team: BDFC (Big Data Fan Club)
#### Members: Sasha Kenkre, Sammy Stern, Yash Kamat, Bulgan Jugderkhuu

This is a capstone project for SI 699 at UMSI Winter 2022.

## Project Goal
With the COVID-19 pandemic, it has become increasingly important for people to wear masks to protect themselves and others from catching the virus.Our goal is to create a model to detect if people are Wearing masks or Not wearing masks/wearing masks improperly.

## Potential Customers and Value
Our potential customers are organizations seeking to monitor the masking of individuals such as: 

- Government agencies

- Business owners

- Airlines

This projects aims to increase public health safety, compliance with mask mandates, and reduce reduces costs in time, effort, money for businesses by preventing a COVID outbreak.


## Data Sources
* [Face Mask Detection | YOLO Darknet Format](https://www.kaggle.com/datasets/parot99/face-mask-detection-yolo-darknet-format)

## Project Architecture
In this project, we used our Kaggle data source for all of the images required for building our model and for training and testing. We then used S3 bucket to host these images on the Cloud without having to save them locally. Next, we connected to AWS Sagemaker to create a Jupyter Notebook instance to build and run our model. Consequently, we connected back to our S3 buckets and used our outputs to connect to create our Athena database.

## The Process
We used MTCNN and Haarcascade for face detection and compared each with our annotated data. If the numbers matched or were close, we used that model for mask detection.

## Results
|             | Face Detection | Proper Mask Detection | Improper/ No Mask Detection    |
| ----------------| ----------------|------------------------|---------------------------|
| Dataset 1 (n=250)     | 78%    |61.2%   |76.4%   |
| Dataset 2 (n=4830)  | 74.8%       |52%    |77.5%    |
| Dataset 2 (± 1) | 74.8%|84%   | 92.7%  |

## Required Packages to Import
```
import os
import numpy as np
import pandas as pd
import boto3
from mtcnn.mtcnn import MTCNN
import PIL
from PIL import Image, ImageOps
from numpy import asarray
from os import listdir
from os.path import isfile, join
import cv2
import requests
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import random
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow
%matplotlib inline
```

### Haarcascade from files
- haarcascade_frontalface_default.xml
- haarcascade/haarcascade_profileface.xml
- haarcascade_nose.xml
- haarcascade/haarcascade_mouth.xml
