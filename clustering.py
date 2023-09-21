import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import DBSCAN

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
