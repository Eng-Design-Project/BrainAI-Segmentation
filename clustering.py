# "# ADJUSTABLE:" indicates all the values that can be changed
# ROI: region of interest

# All additional necessary libraries are in requirements.txt
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from scipy.ndimage import gaussian_filter
from scipy.cluster.hierarchy import dendrogram, linkage
from skimage import morphology, measure, feature, exposure
from skimage.filters import gaussian, median, threshold_local, sobel
from skimage.morphology import ball
import data

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

# SITK TO PYDICOM - MD
# original:
'''
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
'''


#moved to bottom, just to keep sep from everything else
def execute_whole_clustering(input, algo):
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

def tester_algo(input_array):
    """
    input: np array
    prints
    output: format for the output of any clustering algo
    """
    print("clustering testing algo")
    test_coords = [[x, y, z] for x in range(0, 30) for y in range(0, 30) for z in range(0, 30)]
    return test_coords


def execute_seg_clustering(input, algo):
    """
    input: an pre-atlas segmented scan (dict of 3d np arrays) and a string representing the chosen algo
    selects the algo from a dictionary of corresponding functions
    output: a dictionary of region : voxel coordinate lists
    """
    # initialize dictionary to store output
    output_coords = {}
    algos_dict = {
        'test': tester_algo
    } 
    
    for region, scan in input.items():
        output_coords[region] = algos_dict[algo](scan)
        
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

'''
## DBSCAN WITH ATLAS ##

def upload_segments(directory):
    segments = []
    for s in os.listdir(directory):
        filepath = os.path.join(directory, s)
        if os.path.isfile(filepath):  # Ensure the path is a file
            dataset = pydicom.dcmread(filepath, force=True)
            segments.append(dataset)
    return segments

def pixel_data(segments):
    return np.stack([s.pixel_array for s in segments])
    #why not use the function in data for getting 3d np arrays from directory?

def preprocess_seg(images):
    filtered_images = gaussian_filter(images, sigma=1)
    edges = sobel(filtered_images)
    return edges

def apply_thresholding(image):
    block_size = 35
    adaptive_thresh = threshold_local(image, block_size, offset=5)
    binary_adaptive = image > adaptive_thresh
    return binary_adaptive

def dbscan_with_atlas(image_slice):
    coords = np.column_stack(np.where(image_slice > 0))
    db_atl = DBSCAN(eps=1.5, min_samples=7).fit(coords)
    
    # Calculate cluster centers
    cluster_centers = []
    labels = db_atl.labels_
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label != -1:  # Exclude noise label
            members = coords[labels == label]
            center = members.mean(axis=0)
            cluster_centers.append(center)
    
    return db_atl, np.array(cluster_centers)

def get_coordinates(db_atl, labels):
    unique_labels = np.unique(labels)
    cluster_coords = {label: np.column_stack(np.where(labels == label)) for label in unique_labels if label != -1}
    return cluster_coords

def main(directories):
    all_cluster_coords = []
    
    for directory in directories:
        segments = upload_segments(directory)
        images = pixel_data(segments)
        
        for img_slice in images:
            preprocessed_slice = preprocess_seg(img_slice)
            thresholded_slice = apply_thresholding(preprocessed_slice)
            
            db, cluster_centers = dbscan_with_atlas(thresholded_slice)
            cluster_coords = get_coordinates(db, db.labels_)
            
            all_cluster_coords.append(cluster_coords)
    
    return all_cluster_coords

if __name__ == "__main__":
    directories = ["/content/brain", "/content/skull"]
    cluster_coords = main(directories)

    print("Number of Clusters:", len(cluster_coords))
    print("\n3D Coordinates of Clusters:")
    for clusters in cluster_coords:
        for label, coords in clusters.items():
            print(f"{label} :", coords.tolist())
            print()


## K-MEANS ##

def load_volume(directory):
    """Load DICOM slices from a directory and create a 3D volume."""
    slices = [pydicom.dcmread(os.path.join(directory, s)) for s in os.listdir(directory)]
    slices.sort(key=lambda x: int(x.filename.split('_')[1].split('.')[0]))
    volume = np.stack([s.pixel_array for s in slices])
    return volume

def apply_gaussian_filter(volume, sigma=1):
    return gaussian(volume, sigma=sigma)

def apply_median_filter(volume):
    return median(volume, footprint=ball(1))

def preprocess_volume(volume):
    volume = apply_gaussian_filter(volume)
    return apply_median_filter(volume)

def combine_volumes(volume1, volume2):
    return np.concatenate([volume1, volume2])

def kmeans_clustering(volume, n_clusters=4, n_init='auto', max_iter=1000):
    reshaped_volume = volume.reshape((-1, 1))
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=n_init, max_iter=max_iter)
    labels = kmeans.fit_predict(reshaped_volume)
    return labels.reshape(volume.shape), kmeans.cluster_centers_

def calculate_brightness(cluster_centers):
    avg_brightness = {}
    for label, center in enumerate(cluster_centers):
        avg_brightness[label] = center[0]
    return avg_brightness

def extract_coordinates(labeled_volume):
    coordinates = {}
    for label in np.unique(labeled_volume):
        coords = np.argwhere(labeled_volume == label)
        coordinates[label] = coords
    return coordinates

def main():
    brain_dir = input("Enter the path for the brain directory: ")
    skull_dir = input("Enter the path for the skull directory: ")

    brain_volume = preprocess_volume(load_volume(brain_dir))
    skull_volume = preprocess_volume(load_volume(skull_dir))
    
    combined_volume = combine_volumes(brain_volume, skull_volume)
    labeled_volume, cluster_centers = kmeans_clustering(combined_volume)
    
    coordinates = extract_coordinates(labeled_volume)
    avg_brightness = calculate_brightness(cluster_centers)

    return coordinates, labeled_volume, avg_brightness

coordinates, labeled_3d, avg_brightness = main()

print("Average Cluster Brightness:")
for key, value in avg_brightness.items():
    print(key, ":", value)

print("\n3D Coordinates of Clusters:")
for key, value in coordinates.items():
    print(key, ":", value, "\n")



## Hierarchical ##

def load_volume(directory):
    slices = [pydicom.dcmread(os.path.join(directory, s)) for s in os.listdir(directory)]

    # Extract numerical value from the filename and use it for sorting
    slices.sort(key=lambda x: int(x.filename.split('_')[1].split('.')[0]))

    volume = np.stack([s.pixel_array for s in slices])
    return volume


def preprocess_volume(volume):
    scaler = StandardScaler()
    standardized_volume = scaler.fit_transform(volume.reshape(-1, 1)).reshape(volume.shape)
    return standardized_volume

def extract_features(volume):

    # Intensity feature
    intensity = volume.flatten()
    print(f"Intensity shape: {intensity.shape}")

    # Gradient feature
    grad_x, grad_y, grad_z = np.gradient(volume)
    print(f"grad_x shape: {grad_x.flatten().shape}")
    print(f"grad_y shape: {grad_y.flatten().shape}")
    print(f"grad_z shape: {grad_z.flatten().shape}")

    combined_features = np.stack([
        intensity,
        grad_x.flatten(),
        grad_y.flatten(),
        grad_z.flatten()
    ], axis=-1)

    return combined_features


def perform_clustering(features, n_clusters):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward').fit(features)
    return clustering.labels_

def extract_cluster_coordinates(labels, n_clusters):
    clusters_coordinates = {}
    for cluster in range(n_clusters):
        clusters_coordinates[cluster] = np.argwhere(labels == cluster)
    return clusters_coordinates

# Load, preprocess, extract
brain_volume = load_volume('/content/brain')  # path to brain dir
skull_volume = load_volume('/content/skull')  # path to skull dir

print(f"Shape of brain_volume: {brain_volume.shape}")
print(f"Shape of skull_volume: {skull_volume.shape}")

if brain_volume.shape != skull_volume.shape:
    raise ValueError("Brain and Skull volumes aren't the same. Make sure they both have the same # of slices.")

combined_volume = np.concatenate([brain_volume, skull_volume])
combined_volume = preprocess_volume(combined_volume)
features = extract_features(combined_volume)

Z = linkage(features, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.show()


n_clusters = int(input("Number of clusters: "))
labels = perform_clustering(features, n_clusters)
labels_volume = labels.reshape(combined_volume[1:-1].shape)

clusters_coordinates = extract_cluster_coordinates(labels_volume, n_clusters)

#Dustin:
#The main functions bundle helper functions (like pixel_data) and 
# the actual clustering algo (like dbscan_with_atlas), which is fine,
# but it needs to take an np array(3d volume) is input for it to be usable
# by the universal "execute clustering" function(s)
#all the "main" functions should be labeled so they can be implemented
#it also seems like you made many helper functions that do the same thing: loading a volume, normalizing, etc
# clustering shouldn't need to access any directories

'''
