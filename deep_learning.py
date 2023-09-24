import SimpleITK as sitk
import os
import tensorflow as tf
import numpy as np
from scipy.ndimage import convolve
import data


def print_hello():
    print(" Entered deep learning ")

#sample input
sitk_images_dict_hardcode = {
    "image1": data.get_3d_image("scan1"),
    "image2": data.get_3d_image("scan2"),   
    # Add other images...
}

#normalizes pixel values in np array, input nparray dict output nparray dict
def normalizeTF(volume3dDict):
    normalizedDict = {}
    
    for key, value in volume3dDict.items():
        # Convert the value to a tensor
        tensor = tf.convert_to_tensor(value, dtype=tf.float32)
        
        # Find min and max values for the current tensor
        minVal = tf.reduce_min(tensor)
        maxVal = tf.reduce_max(tensor)
        
        # Normalize the tensor
        normalizedTensor = (tensor - minVal) / (maxVal - minVal)
        
        # Convert back to numpy array and store it in the dictionary
        normalizedDict[key] = normalizedTensor.numpy()
        """normalizedDict[key] = normalizedTensor"""
    return normalizedDict

#standard binary classification model
def buildModel(inputShape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(inputShape=inputShape),
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

#pix by pix classification, using window. not built with feedback
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

#wrapper function, gets nparray dict from sitk dict, normalizes, gets shape,
#  puts in basic classif model, doesn't run training
def dlAlgorithm(segmentDict):
    numpyImagesDict = {key: sitk.GetArrayFromImage(img) for key, img in segmentDict.items()}
    normalizedDict = normalizeTF(numpyImagesDict)

    """Currently using 3D arrays, might switch to tensors. In such case, the shape might change."""
    sampleShape = numpyImagesDict[list(normalizedDict.keys())[0]].shape
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



class DeepLearningModule:

    def __init__(self):
        pass

    def load_regions(self, region_data):
        for region_name, sitk_name in region_data.items():
            try:
                region_image = sitk.ReadImage(sitk_name)
                print(f"Loaded {region_name} from {sitk_name}")
            except Exception as e:
                print(f"Error loading {region_name} from {sitk_name}: {e}")

    def load_atlas_data(self, atlas_data1, atlas_data2):
        """
        Load or set the atlas segmentation data from two folders.

        Args:
            atlas_data1: Path to the first atlas segmentation folder.
            atlas_data2: Path to the second atlas segmentation folder.
        """
        global atlas_segmentation_data  # Assuming this is declared globally
        
        atlas_segmentation_data = {}
        
        for folder_path in [atlas_data1, atlas_data2]:
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    atlas_image = sitk.ReadImage(file_path)
                    atlas_segmentation_data[filename] = atlas_image
                    print(f"Loaded atlas data from {file_path}")
                except Exception as e:
                    print(f"Error loading atlas data from {file_path}: {e}")

# Existing user score global variables and function
user_score1 = -1
user_score2 = -2

def get_user_score(x1, x2):
    global user_score1, user_score2
    user_score1 = x1
    user_score2 = x2
    print("score 1 is: ", user_score1)
    print("score 2 is: ", user_score2)