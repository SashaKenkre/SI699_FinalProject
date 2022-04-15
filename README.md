# SI 699 Final Project: Mask Detection
### Team: BDFC (Big Data Fan Club)
#### Members: Sasha Kenkre, Sammy Stern, Yash Kamat, Bulgan Jugderkhuu

This is a capstone project for SI 699 at UMSI Winter 2022.

## Project Goal
With the COVID-19 pandemic, it has become increasingly important for people to wear masks to protect themselves and others from catching the virus. Our goal is to create a model to detect if people are Wearing masks or Not wearing masks/wearing masks improperly.

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
### Cloud Data
We used Vocareum/AWS to host our images and csv files. All our images were read into our Jupyter notebooks with boto3 to connect to our S3 bucket. We used Athena as our final database through which we used SQL queries to calculate our accuracies.

### Face Detection
Our Kaggle data came with annotations for images which provided the number of people in the image wearing a mask and not wearing a mask.

We used MTCNN and Haarcascade for face detection. We first ran MTCNN and checked if the number of faces found matched the number of people from annotations, if not, we used Haarcascade. Then, we had a list of conditionals based on the face count from MTCNN and Haarcascade to see which was more accurate and would be used for mask detection. We saved that value and the model used for face detection in our dataframe. If the predicted number of faces were equal to the actual number of faces, we assigned 1 to `corr_pred`, else 0.

![Face Detect](https://github.com/bulgaan/SI699_FinalProject/blob/main/images/face_detect.png)

### Mask Detection
Next, we moved to mask detection where we used Haarcascades to find the nose and mouth in each image. The face would be detected using the above dataframe based on what was in the `detect_type` column, then we would move to finding the nose and mouth within the found faces. If a nose and mouth were found, the count for no mask increased, and if no nose or mouth were found we increased the count for proper mask. Again, this was added into our dataframe, and if the predicted number of proper/improper masks were equal to the actual numbers, we assigned 1 to `corr_pred`, else 0.

![Mask Detect](images/mask_detect.png?raw=true)

### Accuracy
Once we had these values, we found our model’s accuracy for both mask on and no mask. We took the sum of our correct mask predictions and divided by the total number of images. Then we did the same for no mask prediction. We also did this for + or - 1 because some of the predictions were off by 1. We had a higher accuracy when we did + or - 1.

![Accuracy](https://github.com/bulgaan/SI699_FinalProject/blob/main/images/accuracy.png?raw=true)


## Results
|             | Face Detection | Proper Mask Detection | Improper/ No Mask Detection    |
| ----------------| ----------------|------------------------|---------------------------|
| Dataset 1 (n=250)     | 78%    |61.2%   |76.4%   |
| Dataset 2 (n=4830)  | 74.8%       |52%    |77.5%    |
| Dataset 2 (± 1) | 74.8%|84%   | 92.7%  |

## Required Packages to Import

- `boto3`
- `cv2`
- `matplotlib`
- `mtcnn`
- `numpy`
- `os`
- `pandas`
- `PIL`
- `random`
- `requests`

## Required Haarcascade files
- `haarcascade_frontalface_default.xml`
- `haarcascade_profileface.xml`
- `haarcascade_nose.xml`
- `haarcascade_mouth.xml`

Haarcascade files sourced from [this GitHub repository](https://github.com/Poojan66/facedetection).
