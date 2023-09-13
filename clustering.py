#THIS IS JUST BRAINSTORMING; MOST OF THE CODE IS FROM ONLINE RESOURCES TO TEST OUT THE BEST METHODS BEFORE MOVING FORWARD
'''
import os
import numpy as np
import SimpleITK as sitk
from sklearn.cluster import KMeans
import data

#K-MEANS CLUTERING:

def preprocess_images(image_list):
    # Preprocess images into a suitable format for clustering
    # You might need to resize, normalize, or flatten the images
    processed_images = []
    for image in image_list:
        # Preprocess each image and append to processed_images list
        processed_image = preprocess_image(image)
        processed_images.append(processed_image)
    return processed_images

def preprocess_image(image):
    # Implement preprocessing steps for a single image
    # Example: resizing, normalization, etc.
    # Return the preprocessed image
    return image

# Load your images here using the get_3d_image function from your data module
image_directory = "path_to_your_image_directory"
image_list = data.get_3d_image(image_directory)

# Preprocess the images for clustering
preprocessed_images = preprocess_images(image_list)

# Convert the list of images to a numpy array
image_data = np.array(preprocessed_images)

# Reshape the image data to 2D if necessary (e.g., for grayscale images)
num_samples, image_height, image_width = image_data.shape
image_data_reshaped = image_data.reshape(num_samples, image_height * image_width)

# Perform K-Means clustering
num_clusters = 3  # Specify the number of clusters
kmeans = KMeans(n_clusters=num_clusters)
cluster_labels = kmeans.fit_predict(image_data_reshaped)

# Perform post-processing and visualization based on your needs
# You can assign cluster labels to the original images and visualize the results

# For example, you can iterate through the images and cluster labels
for i, image in enumerate(image_list):
    cluster_label = cluster_labels[i]
    # Perform actions based on cluster label, like saving to cluster-specific folders



#AGGLOMERATIVE CLUSTERING:

import os
import numpy as np
import SimpleITK as sitk
from sklearn.cluster import AgglomerativeClustering
import data

def preprocess_images(image_list):
    # Preprocess images into a suitable format for clustering
    # You might need to resize, normalize, or flatten the images
    processed_images = []
    for image in image_list:
        # Preprocess each image and append to processed_images list
        processed_image = preprocess_image(image)
        processed_images.append(processed_image)
    return processed_images

def preprocess_image(image):
    # Implement preprocessing steps for a single image
    # Example: resizing, normalization, etc.
    # Return the preprocessed image
    return image

# Load your images here using the get_3d_image function from your data module
image_directory = "path_to_your_image_directory"
image_list = data.get_3d_image(image_directory)

# Preprocess the images for clustering
preprocessed_images = preprocess_images(image_list)

# Convert the list of images to a numpy array
image_data = np.array(preprocessed_images)

# Reshape the image data to 2D if necessary (e.g., for grayscale images)
num_samples, image_height, image_width = image_data.shape
image_data_reshaped = image_data.reshape(num_samples, image_height * image_width)

# Perform Agglomerative clustering
num_clusters = 3  # Specify the number of clusters
agglomerative = AgglomerativeClustering(n_clusters=num_clusters)
cluster_labels = agglomerative.fit_predict(image_data_reshaped)

# Perform post-processing and visualization based on your needs
# You can assign cluster labels to the original images and visualize the results

# For example, you can iterate through the images and cluster labels
for i, image in enumerate(image_list):
    cluster_label = cluster_labels[i]
    # Perform actions based on cluster label, like saving to cluster-specific folders

'''
#It's not linked to other modules cause i was testing specific aspects that were easier to pin point as a standalone file

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os

def sitk_np(image_dict):  #sitk dictionary to numpy array
    np_dictionary = {}
    for key, img in image_dict.items():
        np_dictionary[key] = sitk.GetArrayFromImage(img)
    return np_dictionary

def preprocess_slices(slice_2d):  #take 2d sitk image, return with clusters highlighted
   
    X = np.argwhere(slice_2d > 0)  # ignore background pixels; use higher number than 0 for increased threshold
    dbscan = DBSCAN(eps=5, min_samples=5)  #change eps and min here
    labels = dbscan.fit_predict(X)
    
    processed = np.zeros_like(slice_2d, dtype=np.uint8)
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:  
            color = 255
        else:
            color = label * 40  
        processed[X[labels == label, 0], X[labels == label, 1]] = color
    
    return processed

def display_slices(processed_dict): #output slices

    for key, img_3d in processed_dict.items():
        for z in range(img_3d.shape[0]):
            plt.figure()
            plt.title(f"Slice {z} from {key}")
            plt.imshow(img_3d[z], cmap='nipy_spectral')  
           #plt.imshow(img_3d[z], cmap='nipy_spectral', vmin=0, vmax=0) # sometimes 'jet' cmap works better, increase vmax to limit highlighted clusters
            plt.colorbar()
            plt.show()

def process_images(image_dict):
    
    np_dictionary = sitk_np(image_dict)
    
    processed_dict = {}
    for key, img_3d in np_dictionary.items():
        processed_img_3d = np.zeros_like(img_3d, dtype=np.uint8)
        for z in range(img_3d.shape[0]):
            processed_img_3d[z] = preprocess_slices(img_3d[z])
        processed_dict[key] = processed_img_3d
    
    display_slices(processed_dict)
    
    return processed_dict

def load_sitk_folder(folder_path):  #load a dictionary of sitk images (this loads them from local for easier testing, but can easily be changed)

    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    image_dict = {}
    for image_file in image_files:
        full_path = os.path.join(folder_path, image_file)
        try:
            image = sitk.ReadImage(full_path)
            key = os.path.splitext(image_file)[0]  
            image_dict[key] = image
        except:
            print(f"Failed to load {full_path}")
    return image_dict


folder_path = input("Enter folder path: ")
image_dict = load_sitk_folder(folder_path)
processed_images = process_images(image_dict)
