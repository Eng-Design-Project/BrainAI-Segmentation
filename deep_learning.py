import SimpleITK as sitk
import os
import tensorflow as tf
import numpy as np
from scipy.ndimage import convolve
import data


def print_hello():
    print(" Entered deep learning ")


#dummy input data
sitk_images_dict = {
    "image1": data.get_3d_image("scan1"),
    "image2": data.get_3d_image("scan2"),   
    # Add other images...
}

#normalizes pixel value of 3d array
def normalizeTF(volume3dDict):
    normalizedDict = {}
    for key, value in volume3dDict.items():
        tensor = tf.convert_to_tensor(value, dtype=tf.float32)
        minVal = tf.reduce_min(tensor)
        maxVal = tf.reduce_max(tensor)
        normalizedTensor = (tensor - minVal) / (maxVal - minVal)
        
        # Convert back to numpy and store it in the dictionary
        normalizedDict[key] = normalizedTensor.numpy()
    return normalizedDict


#standard binary classifier, probably not useful for our use-case
def buildModel(inputShape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=inputShape),  # Corrected here
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  # Assumes binary classification
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

#pix by pix classifier, not built with user score in mind
def buildPixelModel(input_shape, window_size=3):
    # Assumes input is a 3D patch of size [window_size, window_size, depth]
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (window_size, window_size), activation='relu', padding='valid'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  # Assumes binary classification
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model        

#wrapper for getting the np arrays from sitkimages, normalizing, getting shape,
# and plugging shape into basic classifier
def dlAlgorithm(segmentDict):
    numpyImagesDict = {key: sitk.GetArrayFromImage(img) for key, img in segmentDict.items()}
    normalizedDict = normalizeTF(numpyImagesDict)

    """Currently using 3D arrays, might switch to tensors. In such case, the shape might change."""
    sampleShape = numpyImagesDict[list(numpyImagesDict.keys())[0]].shape
    model = buildModel((sampleShape[1], sampleShape[2], sampleShape[0]))  # (height, width, channels)

def find_boundary(segment):
    # Define a kernel for 3D convolution that checks for 26 neighbors in 3D
    kernel = np.ones((3, 3, 3))
    kernel[1, 1, 1] = 0
    
    # Convolve with the segment to find the boundary of ROI
    boundary = convolve(segment > 0, kernel) > 0
    
    # Keep only boundary voxels that are non-zero in the original segment
    boundary = boundary & (segment > 0)
    
    return boundary

# Existing user score global variables and function
#will prob be removed, user score will be supplied to dl algo as argument from core
user_score1 = -1
user_score2 = -2

#will be removed, user score updated in core
def get_user_score(x1, x2):
    global user_score1, user_score2
    user_score1 = x1
    user_score2 = x2
    print("score 1 is: ", user_score1)
    print("score 2 is: ", user_score2)