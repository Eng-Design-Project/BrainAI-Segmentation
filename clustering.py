import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from skimage.exposure import equalize_adapthist
from skimage import exposure, img_as_ubyte
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter
from skimage import morphology, measure, feature, exposure
from skimage.filters import gaussian, threshold_local, sobel
import data
import segmentation


###SHARED FUNCTIONS###

def clahe_enhance(volume, kernel_size=(3, 3, 3)):
    """
    - 3d clahe enhancement w sliding window
    - takes np array, input has to be 3d volume
    - outputs enhanced 3d volume array
    """

    # You can apply CLAHE on a per-slice basis if the volume is too large
    enhanced = np.empty_like(volume, dtype=np.uint8)  # change dtype to uint8 if you want 8-bit output
    for i in range(volume.shape[0]):
        # Apply CLAHE to each slice
        slice_enhanced = exposure.equalize_adapthist(volume[i], kernel_size=kernel_size[1:])
        # Rescale the result to the range [0, 255] and convert to 8-bit unsigned integers
        enhanced[i] = img_as_ubyte(slice_enhanced)
    return enhanced


def apply_gaussian_filter(volume, sigma=1):
    """
    filter that applies smoothing to the volume
    """
    return gaussian(volume, sigma=sigma)

def create_non_zero_mask(volume, threshold=0):
    """
    mask from non-zero (black) voxels
    """
    return volume > threshold

# same normalization function from core
def clustering_normalize(arr):
    arr64 = arr.astype(np.float64)

    min_val = np.min(arr64)
    max_val = np.max(arr64)

    # Check if max and min values are the same (to avoid division by zero)
    if max_val - min_val == 0:
        return arr64

    normalized_arr64 = (arr64 - min_val) / (max_val - min_val)
    return normalized_arr64


########################################################


## DBSCAN ##
# from core:
    # run "db_execute" to execute the algorithm
def db_preprocess(volume):
    """
    - call gaussian filter
    - the sobel operator is run on all three axis; it's a tensity-based edge detection algorithm
    - gradient magnitude is just the sum of the three sobel operator results-
        to actually apply the edge detection algo to the full scan
    """

    gaussian_filtered_volume = gaussian_filter(volume, sigma = 2)

    gradient_mag = np.sqrt(np.square(sobel(gaussian_filtered_volume, axis=0)) + 
                           np.square(sobel(gaussian_filtered_volume, axis=1)) + 
                           np.square(sobel(gaussian_filtered_volume, axis=2)))

    return gradient_mag


# Apply DBSCAN to preprocessed data
def db_clustering(volume, eps, min_samples):
    """
    - call the preprocessing function
    - call the normalization function
    - call the non-zero mask
    - set the normalized volume
    - reshape the volume into a 2d array
    - run dbscan clustering (both the eps, and min_samples values are inputted in the core)
    - fit the model and combine data [predict cluster labels for each voxel]
    - apply cluster labels to the full volume
    """
    db_preprocessed_volume = db_preprocess(volume)
    
    db_normalized_volume = clustering_normalize(db_preprocessed_volume)

    non_zero_mask = create_non_zero_mask(db_normalized_volume)

    db_normalized_volume = np.maximum(db_normalized_volume, 0)

    reshaped_volume = db_normalized_volume[non_zero_mask].reshape(-1, 1)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    labels = dbscan.fit_predict(reshaped_volume)

    full_volume_labels = np.full(volume.shape, -1, dtype=int)

    full_volume_labels[non_zero_mask] = labels

    return full_volume_labels, dbscan.components_


def db_calculate_brightness(db_cluster_centers, db_labeled_volume):
    avg_brightness = {}
    for label in np.unique(db_labeled_volume):
        if label != -1:  # Exclude noise points
            cluster_voxels = db_labeled_volume == label
            avg_brightness[str(label)] = np.mean(db_cluster_centers[cluster_voxels])
    return avg_brightness


def db_extract_coordinates(db_labeled_volume):
    """
    - initialize library to store coordinates
    - iterate through each unique label
    - exclude noise points
    - find all coordinates that apply to current cluster
    - convert labels to strings and store in the dictionary
    """
    db_coordinates = {}
    for label in np.unique(db_labeled_volume):
        if label != -1:  
            db_coords = np.argwhere(db_labeled_volume == label)
            db_coordinates[str(label)] = db_coords
    return db_coordinates


# Execute DBSCAN clustering
def db_execute(volume, n_clusters, eps, min_samples):
    """
    this is the function that's called when the user wants to run dbscan:
    - preprocess
    - normalize
    - run dbscan clustering
    - extract cluster coordinates
    - calculate avg brightness of cluster
    """
    db_preprocessed_volume = db_preprocess(volume)
    
    db_normalized_volume = clustering_normalize(db_preprocessed_volume)
    
    db_labeled_volume, db_cluster_centers = db_clustering(db_normalized_volume, eps=eps, min_samples=min_samples)
    
    db_coordinates = db_extract_coordinates(db_labeled_volume)
    
    avg_brightness = db_calculate_brightness(db_cluster_centers, db_labeled_volume)
    
    return db_coordinates, db_labeled_volume, avg_brightness

# possible additional future output - not currently used
def db_output(db_coordinates, avg_brightness):
    db_result = "Average Cluster Brightness:\n"
    for key, value in avg_brightness.items():
        db_result += f"{key} : {value}\n"

    db_result += "\n3D Coordinates of Clusters:\n"
    for key, value in db_coordinates.items():
        db_result += f"{key} : {value}\n"
    return db_result



## K-MEANS ##
# from core:
    # run "km_execute" to execute the algorithm

def km_preprocess(volume):
    """
    - call gaussian filter
    - call 3d clahe enhancement function
    """

    gaussian_filtered_volume = gaussian_filter(volume, sigma=1)

    clahe_enhanced_volume = clahe_enhance(gaussian_filtered_volume)

    return clahe_enhanced_volume 


# apply kmeans to preprocessed data
def km_clustering(volume, n_clusters, n_init='auto', max_iter=1000):
    """
    - call the preprocessing function
    - call the normalization function
    - call the non-zero mask
    - set the normalized volume
    - reshape the volume into a 2d array
    - run kmeans clustering (the n_clusters value is inputted in the core)
    - fit the model and combine data [predict cluster labels for each voxel]
    - apply cluster labels to the full volume
    """

    km_preprocessed_volume = km_preprocess(volume)

    km_normalized_volume = clustering_normalize(km_preprocessed_volume)

    non_zero_mask = create_non_zero_mask(km_normalized_volume)

    km_normalized_volume = np.maximum(km_normalized_volume, 0)

    reshaped_volume = km_normalized_volume[non_zero_mask].reshape(-1, 1)

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=n_init, max_iter=max_iter)

    labels = kmeans.fit_predict(reshaped_volume)

    full_volume_labels = np.full(volume.shape, -1)
    full_volume_labels[non_zero_mask] = labels

    return full_volume_labels, kmeans.cluster_centers_


def km_calculate_brightness(km_cluster_centers):
    avg_brightness = {}
    for label, center in enumerate(km_cluster_centers):
        avg_brightness[str(label)] = center[0]
    return avg_brightness

def km_extract_coordinates(km_labeled_volume):
    """
    - initialize library to store coordinates
    - iterate through each unique label
    - find all coordinates that apply to current cluster
    - convert labels to strings and store in the dictionary
    """
    km_coordinates = {}
    for label in np.unique(km_labeled_volume):
        km_coords = np.argwhere(km_labeled_volume == label)
        km_coordinates[str(label)] = km_coords
    return km_coordinates


def km_execute(volume, n_clusters, eps, min_samples):
    """
    this is the function that's called when the user wants to run kmeans:
    - preprocess
    - normalize
    - run kmeans clustering
    - extract cluster coordinates
    """
    km_preprocessed_volume = km_preprocess(volume)
    km_normalized_volume = clustering_normalize(km_preprocessed_volume)
    km_labeled_volume, km_cluster_centers = km_clustering(km_normalized_volume, n_clusters=n_clusters)
    km_coordinates = km_extract_coordinates(km_labeled_volume)
    return km_coordinates, km_labeled_volume, km_calculate_brightness(km_cluster_centers)

# possible additional future output - not currently used
def km_output(km_coordinates, avg_brightness):
    km_result = "Average Cluster Brightness:\n"
    for key, value in avg_brightness.items():
        km_result += f"{key} : {value}\n"

    km_result += "\n3D Coordinates of Clusters:\n"
    for key, value in km_coordinates.items():
        km_result += f"{key} : {value}\n"

    return km_result


## Hierarchical - SLINK##
# from core:
    # run "sl_execute" to execute the algorithm

def sl_preprocess(volume):
    """
    - call gaussian filter
    - call 3d clahe enhancement function
    - morphological opening - removes white spots from foreground
    """
    gaussian_filtered_volume = gaussian_filter(volume, sigma=1.5)

    clahe_enhanced_volume = clahe_enhance(gaussian_filtered_volume, kernel_size=(2, 2, 2))

    opened_volume = morphology.opening(clahe_enhanced_volume, morphology.ball(2))

    return opened_volume

def slink_clustering(volume, n_clusters):
    """
    - call the preprocessing function
    - call the normalization function
    - call the non-zero mask
    - set the normalized volume
    - reshape the volume into a 2d array
    - run single-linkage agglomerative clustering (the n_clusters value is inputted in the core)
    - fit the model and combine data [predict cluster labels for each voxel]
    - apply cluster labels to the full volume
    """
    sl_preprocessed_volume = sl_preprocess(volume)

    sl_normalized_volume = clustering_normalize(sl_preprocessed_volume)

    non_zero_mask = create_non_zero_mask(sl_normalized_volume)

    sl_normalized_volume = np.maximum(sl_normalized_volume, 0)

    reshaped_volume = sl_normalized_volume[non_zero_mask].reshape(-1, 1)

    # Single Linkage Hierarchical Clustering
    slink = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')

    labels = slink.fit_predict(reshaped_volume)

    full_volume_labels = np.full(volume.shape, -1)
    full_volume_labels[non_zero_mask] = labels

    return full_volume_labels

def sl_calculate_brightness(sl_labeled_volume):
    avg_brightness = {}
    unique_labels = np.unique(sl_labeled_volume)
    for label in unique_labels:
        avg_brightness[str(label)] = np.mean(sl_labeled_volume[sl_labeled_volume == label])
    return avg_brightness

def sl_extract_coordinates(sl_labeled_volume):
    """
    - initialize library to store coordinates
    - iterate through each unique label
    - find all coordinates that apply to current cluster
    - convert labels to strings and store in the dictionary
    """
    sl_coordinates = {}
    for label in np.unique(sl_labeled_volume):
        sl_coords = np.argwhere(sl_labeled_volume == label)
        sl_coordinates[str(label)] = sl_coords
    return sl_coordinates

def sl_execute(volume, n_clusters, eps, min_samples):
    """
    this is the function that's called when the user wants to run hierarchical:
    - preprocess
    - normalize
    - run hierarchical clustering
    - extract cluster coordinates
    """
    sl_preprocessed_volume = sl_preprocess(volume)
    sl_normalized_volume = clustering_normalize(sl_preprocessed_volume)
    sl_labeled_volume = slink_clustering(sl_normalized_volume, n_clusters=n_clusters)
    sl_coordinates = sl_extract_coordinates(sl_labeled_volume)
    return sl_coordinates, sl_labeled_volume, sl_calculate_brightness(sl_labeled_volume)

# possible additional future output - not currently used
def sl_output(sl_coordinates):
    sl_result = "Average Cluster Brightness:\n"
    for key, value in avg_brightness.items():
        sl_result += f"{key} : {value}\n"

    sl_result = "\n3D Coordinates of Clusters:\n"
    for key, value in sl_coordinates.items():
        sl_result += f"{key} : {value}\n"
    return sl_result


########################################################

def convert_to_lists(dict_of_arrays):
    dict_of_lists = {}
    for key, array in dict_of_arrays.items():
        # Swap the first and third columns (axis)
        array[:, [0, 2]] = array[:, [2, 0]]

        # Convert each 2D array into a list of lists
        list_of_lists = array.tolist()
        dict_of_lists[key] = list_of_lists
    return dict_of_lists

def execute_whole_clustering(input, algo, n_clusters, eps, min_samples):
    """
    input: an entire scan (3d np array) and a string representing the chosen algo
    selects the algo from a dictionary of corresponding functions
    output: a dictionary of region : voxel coordinate lists
    """
     # initialize dictionary to store output
    output_coords = {}

    #dict of strings that correspond to functions
    algos_dict = {
        'DBSCAN': db_execute,
        'K-Means': km_execute,
        'Hierarchical': sl_execute
    } 

    output_coords, labeled_volume, avg_brightness = algos_dict[algo](input, n_clusters, eps, min_samples)
    
    output_coords = convert_to_lists(output_coords)
    
    return output_coords



def execute_seg_clustering(input, algo, n_clusters, eps, min_samples):
    """
    input: an pre-atlas segmented scan (dict of 3d np arrays) and a string representing the chosen algo
    selects the algo from a dictionary of corresponding functions
    output: a dictionary of region : voxel coordinate lists
    """
    # initialize dictionary to store output
    output_coords_dict = {}

    #dict of strings that correspond to functions
    algos_dict = {
        'DBSCAN': db_execute,
        'K-Means': km_execute,
        'Hierarchical': sl_execute
    }
    
    for region, scan in input.items():
        output_coords, labeled_volume, avg_brightness = algos_dict[algo](scan, n_clusters, eps, min_samples)
        output_coords = convert_to_lists(output_coords)
        output_coords_dict[region] = output_coords

    
        
    return output_coords_dict


if __name__ == "__main__":
    #folder_path = input("Enter folder path: ") # get folder
    folder_path = "scan1"
    volume = data.get_3d_image(folder_path) # create 3d volume

    coordinates = execute_whole_clustering(volume, "DBSCAN", 2)

    clustered_dict = segmentation.create_seg_images_from_image(volume, coordinates)
    data.display_seg_np_images(clustered_dict)



def tester_algo(input_array):
    """
    input: np array
    prints
    output: format for the output of any clustering algo
    """
    print("clustering testing algo")
    test_coords = [[x, y, z] for x in range(0, 30) for y in range(0, 30) for z in range(0, 30)]
    return test_coords


def execute_whole_clustering_old(input, algo):
    """
    input: an entire scan (3d np array) and a string representing the chosen algo
    selects the algo from a dictionary of corresponding functions
    output: a dictionary of region : voxel coordinate lists
    """
     # initialize dictionary to store output
    output_coords = {}

    #dict of strings that correspond to functions
    algos_dict = {
        'dbscan_3d': dbscan_3d
    }

    # perform dbscan and get labeled volume, coordinates, and binary masks for each slice in the output dictionary
    labeled_volume, cluster_coords, brain_mask, skull_mask = algos_dict[algo](input)

    # determine coordinates
    brain_cluster_coordinates, skull_cluster_coordinates = cluster_coordinates(cluster_coords, brain_mask, skull_mask)

    # dictionary to store output coordinates
    output_coords["skull"] = skull_cluster_coordinates
    #display_slices(volume, labeled_volume, cluster_coords, brain_mask, skull_mask)
    #dbscan optimized for entire brain, not atlas segments, currently outputs brain coords as opposed to "skull coords"

    return output_coords


## DBSCAN WITHOUT ATLAS ##

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



