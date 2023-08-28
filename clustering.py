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