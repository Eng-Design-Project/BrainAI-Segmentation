#All necessary libraries are in requirements.txt
import numpy as np
import matplotlib.pyplot as plt
import os
import pydicom
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage import measure
from skimage import feature
from skimage import exposure
from skimage.filters import threshold_local

#dictionary of dcm images as input
def input_dcm_dict(folder_path):
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    slices = [pydicom.dcmread(os.path.join(folder_path, f)) for f in image_files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return np.stack([s.pixel_array for s in slices])

# Gausian probability density to "smooth" and enhance contrast. 
# This decreases noise in the scan.
def apply_gaussian_filter(roi_volume):
    return gaussian_filter(roi_volume, sigma=1)

#also helps with smoothing aka decreasing noise
def apply_clahe(smoothed_roi_volume):
    return exposure.equalize_adapthist(smoothed_roi_volume)

#Adaptive Thresholding
#Block size and offset values are not finalized. 
# I'm still testing different values to determine what's best.
def adaptive_thresholding(clahe_roi_volume):
    adaptive_thresh = threshold_local(clahe_roi_volume, block_size=35, offset=0.1)
    return clahe_roi_volume > adaptive_thresh

#using density-based clustering
#eps and min are not 100% accurate yet.
#I'm still learning how to accurately do the necessary histogram and k-distance graph to accurately predict the epsilon.
#min samples are heavily dependent on noise, so as more smoothing, regularization, and filtering functions are used, the lower the number will get.
def dbscan_clustering(X):
    return DBSCAN(eps=2, min_samples=4).fit_predict(X)

def preprocess_3d(volume):
   
    # Set an ROI (region of interest) to have parameters for the brain's position in each slice. 
    # Focusing on a smaller area gives the function a chance to be more thorough/accurate.
    x_min, x_max = 25, 105
    y_min, y_max = 20, 110
    roi_volume = volume[:, y_min:y_max, x_min:x_max]

    smoothed_roi_volume = apply_gaussian_filter(roi_volume)
    clahe_roi_volume = apply_clahe(smoothed_roi_volume)

    #The binary masks helps identify the lighter segments. 
    #This way the black background wouldn't be a part of the mask
    binary_mask = adaptive_thresholding(clahe_roi_volume)

    # Performing clustering only on the masked region of ROI
    X = np.column_stack(np.nonzero(binary_mask))
    roi_labels = dbscan_clustering(X)

    labeled_volume = np.full_like(volume, fill_value=-1, dtype=np.int32)
    for i, coord in enumerate(X):
        labeled_volume[coord[0], y_min + coord[1], x_min + coord[2]] = roi_labels[i]

    cluster_coords = {label: X[roi_labels == label] for label in set(roi_labels) if label != -1}

    brain_mask, skull_mask = initialize_masks(roi_volume)

    #iterating through each of the different slices to ensure the overall process is accurate instead of just applying it once to the 3d image.
    for idx in range(roi_volume.shape[0]):
        slice_clahe = clahe_roi_volume[idx]
        edges = feature.canny(slice_clahe)
        closed_edges = morphology.binary_closing(edges)
        labeled_regions, num_regions = measure.label(closed_edges, return_num=True)

        # Identifying Skull and Brain Regions. 
        # These are still not 100% accurate, but they're a lot better
        if num_regions > 1:
            brain_region_label = 1 + np.argmax([np.sum(labeled_regions == i) for i in range(1, num_regions)])
            brain_mask[idx] = labeled_regions == brain_region_label
            skull_mask[idx] = labeled_regions > 0
            skull_mask[idx][brain_mask[idx]] = False

    # Adjusting the brain_mask and skull_mask back to the entire image size
    whole_brain_mask, whole_skull_mask = adjust_masks_to_whole_volume(brain_mask, skull_mask, volume, x_min, x_max, y_min, y_max)

    return labeled_volume, cluster_coords, whole_brain_mask, whole_skull_mask

# Initializing full-size masks
def initialize_masks(roi_volume):
    brain_mask = np.zeros_like(roi_volume, dtype=bool)
    skull_mask = np.zeros_like(roi_volume, dtype=bool)
    return brain_mask, skull_mask

#applying masks to the full scan volume
def adjust_masks_to_whole_volume(brain_mask, skull_mask, volume, x_min, x_max, y_min, y_max):
    whole_brain_mask = np.zeros_like(volume, dtype=bool)
    whole_skull_mask = np.zeros_like(volume, dtype=bool)
    whole_brain_mask[:, y_min:y_max, x_min:x_max] = brain_mask
    whole_skull_mask[:, y_min:y_max, x_min:x_max] = skull_mask
    return whole_brain_mask, whole_skull_mask

#display coordinates of brain and skull clusters in output
def cluster_coordinates(cluster_coords, brain_mask, skull_mask):
    x_min, y_min = 25, 20

    brain_coordinates = {}
    skull_coordinates = {}

    for label, coords in cluster_coords.items():
        coords[:, 1] += y_min
        coords[:, 2] += x_min

        for coord in coords:
            if brain_mask[tuple(coord)]:
                if label not in brain_coordinates:
                    brain_coordinates[label] = []
                brain_coordinates[label].append(coord)
            elif skull_mask[tuple(coord)]:
                if label not in skull_coordinates:
                    skull_coordinates[label] = []
                skull_coordinates[label].append(coord)

    brain_coordinates = {k: np.array(v) for k, v in brain_coordinates.items()}
    skull_coordinates = {k: np.array(v) for k, v in skull_coordinates.items()}

    return brain_coordinates, skull_coordinates

#normalization on each of the individual 2d slices
#this helps ensure more accuracy compared to just tesing the 3d image once
def normalize_slice_image(slice_image):
    rgb_img = np.stack([slice_image] * 3, axis=-1)
    rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))
    return rgb_img

def apply_colors_to_labels(slice_labels, rgb_img):
    unique_labels = np.unique(slice_labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    for label in unique_labels:
        if label != -1:
            mask = slice_labels == label
            rgb_img[mask, :] = colors[label, :3]
    return rgb_img

# setting different colors with the parameters of the brain and skull masks/regions
def apply_roi_masks(rgb_img, roi_brain_mask, roi_skull_mask):
    rgb_img[roi_brain_mask] = [1, 0, 0]  # Brain = Red
    rgb_img[roi_skull_mask] = [0, 1, 0]  # Skull = Green
    return rgb_img

#helps keep original image visible under highlighted clusters. 
#function called and expanded in display_slices()
def display_rgb_image(rgb_img, z):
    plt.figure()
    plt.title(f"Slice {z}")
    plt.imshow(rgb_img)
    plt.show()

#showing the slices in the output after being converted back into original 2d form, and making sure the clusters are highlighted accordingly.
def display_slices(volume, labels, cluster_coords, brain_mask, skull_mask):
    x_min, x_max = 25, 105
    y_min, y_max = 20, 110

    for z in range(volume.shape[0]):
        slice_image = volume[z]
        slice_labels = labels[z]

        rgb_img = normalize_slice_image(slice_image)
        rgb_img = apply_colors_to_labels(slice_labels, rgb_img)

        roi_brain_mask = np.zeros_like(slice_image, dtype=bool)
        roi_skull_mask = np.zeros_like(slice_image, dtype=bool)
        roi_brain_mask[y_min:y_max, x_min:x_max] = brain_mask[z, y_min:y_max, x_min:x_max]
        roi_skull_mask[y_min:y_max, x_min:x_max] = skull_mask[z, y_min:y_max, x_min:x_max]

        rgb_img = apply_roi_masks(rgb_img, roi_brain_mask, roi_skull_mask)

        display_rgb_image(rgb_img, z)

#Used as main sript, this helps a lot with testing and pinpointing errors.
#I'm already working on creating function shortcuts and combining factors for easy use as a sub-module instead.
if __name__ == "__main__":
    folder_path = input("Enter folder path: ")
    volume = input_dcm_dict(folder_path)

    labeled_volume, cluster_coords, brain_mask, skull_mask = preprocess_3d(volume)
    brain_coordinates, skull_coordinates = cluster_coordinates(cluster_coords, brain_mask, skull_mask)
    display_slices(volume, labeled_volume, cluster_coords, brain_mask, skull_mask)

    print("3D Brain Cluster Coordinates:")
    print(brain_coordinates)

    print("3D Skull Cluster Coordinates:")
    print(skull_coordinates) 
