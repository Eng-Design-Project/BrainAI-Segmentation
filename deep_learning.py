import SimpleITK as sitk
import os
import tensorflow as tf
import numpy as np
import data

<<<<<<< Updated upstream
print(" Entered deep learning ")
=======

def print_hello():
    print(" Entered deep learning ")



sitk_images_dict = {
    "image1": data.get_3d_image("scan1"),
    "image2": data.get_3d_image("scan2"),   
    # Add other images...
}


numpy_images_dict = {key: sitk.GetArrayFromImage(img) for key, img in sitk_images_dict.items()}


def normalize_tf(volume_3d):
    tensor = tf.convert_to_tensor(volume_3d, dtype=tf.float32)
    min_val = tf.reduce_min(tensor)
    max_val = tf.reduce_max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor.numpy()

normalized_image = normalize_tf(numpy_images_dict["image1"])

print(normalized_image.min())
print(normalized_image.max())

"""
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
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

"""



>>>>>>> Stashed changes
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

# Global variable for atlas segmentation data
atlas_segmentation_data = {}

# Existing user score global variables and function
user_score1 = -1
user_score2 = -2

def get_user_score(x1, x2):
    global user_score1, user_score2
    user_score1 = x1
    user_score2 = x2
    print("score 1 is: ", user_score1)
    print("score 2 is: ", user_score2)