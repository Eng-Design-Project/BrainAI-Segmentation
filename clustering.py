# "# ADJUSTABLE:" indicates all the values that can be changed
# ROI: region of interest

# All additional necessary libraries are in requirements.txt
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
import data


# Gaussian filter 
# uses the gaussian PDF to smooth/ lower contrast in the roi by blurring and reducing noise
def apply_gaussian_filter(roi_volume):
    return gaussian_filter(roi_volume, sigma=1) 
    # ADJUSTABLE: sigma is the stdev of the gaussian distribution
    # a higher sigma can make it smoother (aka less noise), but will also blur it more which decreases accuracy

# CLAHE: Contrast Limited Adaptive Histogram Equalization 
# enhances local contrast; so it only adjusts exposure in smaller areas of the scan where it's more necessary without jeopardizing quality
def apply_clahe(smoothed_roi_volume):
    return exposure.equalize_adapthist(smoothed_roi_volume) # specifically applies clahe alg from the exposure submodule

# Adaptive Thresholding - evaluates each individual pixel to its surrounding mean
# this creates binary masks where it separates the two areas at the edge where they change
def adaptive_thresholding(clahe_roi_volume):
    adaptive_thresh = threshold_local(clahe_roi_volume, block_size=35, offset=0.1)
    return clahe_roi_volume > adaptive_thresh
    # ADJUSTABLE: block_size is the size of the pixel's local region. offset is subtracted from the mean of the local area
    # higher block size = larger area; higher offset = more lenient aka lowers brightness requirements

# DBSCAN: density-based spatial clustering of applications with noise
# works really well with handling noise, outliers, and finding arbitrary-shaped clusters
def dbscan_clustering(X):
    return DBSCAN(eps=2, min_samples=4).fit_predict(X)
    # ADJUSTABLE: the epsilon is a parameter that determines the radius of the area around a voxel (volume pixel)
    # ADJUSTABLE: min_samples are the minimum number of voxels that are within the radius (eps) of the core point for it to be considered a cluster
    # higher eps = larger radius aka larger/fewer clusters (more lenient); higher min_samples = fewer/denser (more strict)
    # ex: eps=2, min_samples=4. it tests a voxel as a core point, if at least 3 other voxels are in the radius of 2 from it, then it's a cluster.
        # then if at least 3 voxels are within a radius of 2 from any of the points within the original cluster, they're also added.

def dbscan_3d(volume):
   
    # Focusing on a smaller area gives the function a chance to be more thorough/accurate.
    x_min, x_max = 25, 105
    y_min, y_max = 20, 110
    roi_volume = volume[:, y_min:y_max, x_min:x_max]
    # ADJUSTABLE: these are the values of the parameter that determine that area to be tested and ignores the rest of the background

    # call the previous 3 functions
    smoothed_roi_volume = apply_gaussian_filter(roi_volume) 
    clahe_roi_volume = apply_clahe(smoothed_roi_volume)
    binary_mask = adaptive_thresholding(clahe_roi_volume)

    # performing dbscan clustering only on the binary masked region of the ROI
    X = np.column_stack(np.nonzero(binary_mask))
    roi_labels = dbscan_clustering(X)

    labeled_volume = np.full_like(volume, fill_value=-1, dtype=np.int32) # fill_value = -1 gives all the voxels in the input a no-label state, meaning they have no been tested nor are they a part of a pre-determined cluster
    for i, coord in enumerate(X):
        labeled_volume[coord[0], y_min + coord[1], x_min + coord[2]] = roi_labels[i]

    # creating a dict of coordinates for each cluster
    cluster_coords = {label: X[roi_labels == label] for label in set(roi_labels) if label != -1} # sets condition to exclude noise points (if label != -1)

    # initializing binary masks; creates a representation of which parts are considered brain/skull respectively
    brain_mask, skull_mask = initialize_masks(roi_volume)

    # iterating through each slice to identify the true regions of the brain and skull
    for idx in range(roi_volume.shape[0]):
        slice_clahe = clahe_roi_volume[idx] # apply equalization to each slice 
        edges = feature.canny(slice_clahe) # detect and identify edges of where one ends and the other begins (feature submodule of skimage)
        closed_edges = morphology.binary_closing(edges) # detects edges and performs binary closing aka any gaps/holes in the areas within those edges will be filled in so they can all be combined (morphology submodule of skimage)
        labeled_regions, num_regions = measure.label(closed_edges, return_num=True) # label the connected regions, then count how many there are (measure submodule of skimage)

        # Identifying Skull and Brain Regions. 
        if num_regions > 1:
            brain_region_label = 1 + np.argmax([np.sum(labeled_regions == i) for i in range(1, num_regions)]) # finds the largest region within the parameters and labels it as the brain
            brain_mask[idx] = labeled_regions == brain_region_label # creating the binary mask of the brain
            skull_mask[idx] = labeled_regions > 0 # creating the binary mask for the skull (this step will make the mask identify everything as a part of the "skull")
            skull_mask[idx][brain_mask[idx]] = False # subtracting the prelabeled brain area from the overall skull mask, leaving only the skull within the second binary mask

    # Adjusting the brain_mask and skull_mask back to the size of the entire image volume so coordinates aren't measured within only the roi parameters
    whole_brain_mask, whole_skull_mask = adjust_masks_to_whole_volume(brain_mask, skull_mask, volume, x_min, x_max, y_min, y_max)

    return labeled_volume, cluster_coords, whole_brain_mask, whole_skull_mask

# initializing masks with the roi volume (3d array)
# this creates the array and initial shape, but sets all elements in the array to 0 since the actual mask volume values aren't identified yet
# pretty much placeholder arrays that get updated
def initialize_masks(roi_volume):
    brain_mask = np.zeros_like(roi_volume, dtype=bool)
    skull_mask = np.zeros_like(roi_volume, dtype=bool)
    return brain_mask, skull_mask

# initializing masks to the full scan volume
# also creating arrays placeholder arrays
def adjust_masks_to_whole_volume(brain_mask, skull_mask, volume, x_min, x_max, y_min, y_max):
    whole_brain_mask = np.zeros_like(volume, dtype=bool)
    whole_skull_mask = np.zeros_like(volume, dtype=bool)
    whole_brain_mask[:, y_min:y_max, x_min:x_max] = brain_mask
    whole_skull_mask[:, y_min:y_max, x_min:x_max] = skull_mask
    return whole_brain_mask, whole_skull_mask

# display coordinates of brain and skull clusters in output
def cluster_coordinates(cluster_coords, brain_mask, skull_mask):
    x_min, y_min = 25, 20 # ADJUSTABLE: as long as it matches the initial roi values

    # initializing empty dictionaries
    brain_cluster_coordinates = {}
    skull_cluster_coordinates = {}

    # adjusting x and y coordinates to represent the full image, not just the roi
    for label, coords in cluster_coords.items():
        coords[:, 1] += y_min
        coords[:, 2] += x_min

        # iterate through each coordinate
        for coord in coords:
            
            # if a coordinate is in the brain mask, append to brain cluster coordinates
            if brain_mask[tuple(coord)]:
                if label not in brain_cluster_coordinates:
                    brain_cluster_coordinates[label] = []
                brain_cluster_coordinates[label].append(coord)

            # if a coordinate is in the skull mask, append to skull cluster coordinates
            elif skull_mask[tuple(coord)]:
                if label not in skull_cluster_coordinates: # create new list for new/unknown label
                    skull_cluster_coordinates[label] = []
                skull_cluster_coordinates[label].append(coord)

    # convert both sets of coordinates into np arrays
    brain_cluster_coordinates = {k: np.array(v) for k, v in brain_cluster_coordinates.items()}
    skull_cluster_coordinates = {k: np.array(v) for k, v in skull_cluster_coordinates.items()}

    return brain_cluster_coordinates, skull_cluster_coordinates

# normalization on each of the individual 2d slices
# this helps ensure more accuracy compared to just testing the 3d image once
def normalize_slice_image(slice_image):
    rgb_img = np.stack([slice_image] * 3, axis=-1)
    rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))
    return rgb_img

def apply_colors_to_labels(slice_labels, rgb_img):
    unique_labels = np.unique(slice_labels) # find unique labels
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    for label in unique_labels:
        if label != -1:  # ignore background (== -1)
            mask = slice_labels == label    # create a mask for the label
            rgb_img[mask, :] = colors[label, :3]    # apply color to corresponding mask
    return rgb_img

# setting different colors with the parameters of the brain and skull masks/regions
def apply_roi_masks(rgb_img, roi_brain_mask, roi_skull_mask):

    # ADJUSTABLE: can use different colors
    rgb_img[roi_brain_mask] = [1, 0, 0]  # Brain = Red
    rgb_img[roi_skull_mask] = [0, 1, 0]  # Skull = Green
    return rgb_img

def display_rgb_image(rgb_img, z):
    plt.figure()    
    plt.title(f"Slice {z}")  # set title to slice number
    plt.imshow(rgb_img)
    plt.show()

# display 2d slices in output
def display_slices(volume, labels, cluster_coords, brain_mask, skull_mask):
    x_min, x_max = 25, 105
    y_min, y_max = 20, 110

    # iterate through slices in 3d volume
    for z in range(volume.shape[0]):
        slice_image = volume[z]  # extract slice
        slice_labels = labels[z]  # extract label

        rgb_img = normalize_slice_image(slice_image)  # normalize slice   
        rgb_img = apply_colors_to_labels(slice_labels, rgb_img)  # color code slice

        # initialize roi masks for extracted slice
        roi_brain_mask = np.zeros_like(slice_image, dtype=bool)
        roi_skull_mask = np.zeros_like(slice_image, dtype=bool)

        # apply corresponding roi masks to initialized masks
        roi_brain_mask[y_min:y_max, x_min:x_max] = brain_mask[z, y_min:y_max, x_min:x_max]
        roi_skull_mask[y_min:y_max, x_min:x_max] = skull_mask[z, y_min:y_max, x_min:x_max]

        # apply masks to the image
        rgb_img = apply_roi_masks(rgb_img, roi_brain_mask, roi_skull_mask)

        # display full processed image
        display_rgb_image(rgb_img, z)

def execute_clustering(sitk_dict, algo):
    output_coords = {} # initialize sitk dictionary to store output
    algos_dict = {
        'dbscan_3d': dbscan_3d
    } 

    for key in sitk_dict:

        # perform dbscan and get labeled volume, coordinates, and binary masks for each slice in the output dictionary
        labeled_volume, cluster_coords, brain_mask, skull_mask = algos_dict[algo](sitk_dict.key)

        # determine coordinates
        brain_cluster_coordinates, skull_cluster_coordinates = cluster_coordinates(cluster_coords, brain_mask, skull_mask)
        
        # dictionary to store output coordinates
        output_coords[key] = brain_cluster_coordinates
        #display_slices(volume, labeled_volume, cluster_coords, brain_mask, skull_mask)
    #dbscan optimized for entire brain, not atlas segments, currently outputs brain coords as opposed to "skull coords"
    return output_coords

# used as main sript, this helps a lot with testing and pinpointing errors.
#I'm already working on creating function shortcuts and combining factors for easy use as a sub-module instead.
if __name__ == "__main__":
    folder_path = input("Enter folder path: ") # get folder
    volume = data.get_3d_array_from_file(folder_path) # create 3d volume

    # apply dbscan to 3d and get labels, overall coordinates, and binary masks
    labeled_volume, cluster_coords, brain_mask, skull_mask = dbscan_3d(volume)

    # find brain and skull coordinates
    brain_cluster_coordinates, skull_cluster_coordinates = cluster_coordinates(cluster_coords, brain_mask, skull_mask)

    display_slices(volume, labeled_volume, cluster_coords, brain_mask, skull_mask)

    print("3D Brain Cluster Coordinates:")
    print(brain_cluster_coordinates)

    print("3D Skull Cluster Coordinates:")
    print(skull_cluster_coordinates) 
