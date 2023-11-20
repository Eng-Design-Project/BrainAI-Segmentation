# "# ADJUSTABLE:" indicates all the values that can be changed

import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from skimage.exposure import equalize_adapthist
from skimage import exposure, img_as_ubyte
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter
from skimage import morphology, measure, feature, exposure
from skimage.filters import gaussian, threshold_local, sobel
from skimage.feature import graycomatrix, graycoprops
import data
import segmentation

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







## SHARED FUNCTIONS ##


# 3d clahe enhancement w sliding window
# takes np array, input has to be 3d volume
# outputs enhanced 3d volume array



def clahe_enhance(volume, kernel_size=(3, 3, 3)):
    # Assume volume is a 3D array with shape (depth, height, width)
    # You can apply CLAHE on a per-slice basis if the volume is too large
    enhanced = np.empty_like(volume, dtype=np.uint8)  # change dtype to uint8 if you want 8-bit output
    for i in range(volume.shape[0]):
        # Apply CLAHE to each slice
        slice_enhanced = exposure.equalize_adapthist(volume[i], kernel_size=kernel_size[1:])
        # Rescale the result to the range [0, 255] and convert to 8-bit unsigned integers
        enhanced[i] = img_as_ubyte(slice_enhanced)
    return enhanced
# def clahe_enhance(volume, kernel_size=(3, 3, 3)): # 'tuple' kernel size for the windowing operation

#     # get half dimensions for padding and indexing
#     half_depth = kernel_size[0] // 2
#     half_height = kernel_size[1] // 2
#     half_width = kernel_size[2] // 2

#     # input volume gets padded for edge cases
#     padded = np.pad(volume, ((half_depth, half_depth), (half_height, half_height), (half_width, half_width)))
    
#     # intiialize emppty volume for a place to store enhanced data
#     enhanced = np.zeros_like(volume)
    
#     # getting volume dimensions of the input
#     depth, height, width = volume.shape

#     # looping through each voxel in the volume
#     for z in range(depth):
#         for y in range(height):
#             for x in range(width):

#                 # extracting a local block centered at the current voxel
#                 local_block = padded[z:z+2*half_depth+1, y:y+2*half_height+1, x:x+2*half_width+1]

#                 # applying adaptive histogram equalization to the local block
#                 local_enhanced = exposure.equalize_adapthist(local_block)

#                 # assigning the center voxel of enhanced block to corresponding voxel in enhanced volume
#                 enhanced[z, y, x] = local_enhanced[half_depth, half_height, half_width]

#     return enhanced

# GLCM (Gray-Level Co-occurence Matrix)
# computes texture features
# takes np array, input has to be 3d volume
# outputs array of texture features
def texture_features(volume):
    # get input volume dimensions
    depth, height, width = volume.shape
    
    # list to store features
    all_features = []
    
    # loop through each slice in volume
    for i in range(depth):
        slice_ = volume[i, :, :]  # extracting axial slice
        
        # computes GLCM for the current slice
        glcm = graycomatrix(slice_, [1], [0], 256, symmetric=True, normed=True)

        # extracting texture features from the glcm
        features = [
            graycoprops(glcm, 'contrast')[0, 0],    # intensity comparison between neighboring voxels
            graycoprops(glcm, 'dissimilarity')[0, 0],   # just like 'contrast', except less sensitive to big differences in gray level values
            graycoprops(glcm, 'homogeneity')[0, 0], # closeness of elemnt distribution -- this is pretty much the same as the inverse difference moment
            graycoprops(glcm, 'energy')[0, 0],  # this is identical to the 'angular second moment' feature i used in the old spectral algo -- sum of squared elements
            graycoprops(glcm, 'correlation')[0, 0]  # joint probability occurence of joint pairs
        ]

        all_features.append(features)

    return np.array(all_features)

# filter that applies smoothing to the volume
def apply_gaussian_filter(volume, sigma=1):
    return gaussian(volume, sigma=sigma)




## DBSCAN WITH ATLAS ##
# from core:
    # run "db2_execute" to execute the algorithm
    # call "db2_output" to display output
def db2_clahe_enhance(volume, kernel_size=(3, 3, 3)):
    # You can apply CLAHE on a per-slice basis if the volume is too large
    enhanced = np.empty_like(volume, dtype=np.float64)
    for i in range(volume.shape[0]):
        enhanced[i] = equalize_adapthist(volume[i], kernel_size=kernel_size[1:])
    return enhanced


def db2_preprocess(volume):

    # apply gaussian
    gaussian_filtered_volume = gaussian_filter(volume, sigma=1)

    # computing gradient magnitude w the sobel filter
    # this finds the intensity rate of change and helps with highlighting edges
    gradient_mag = np.sqrt(np.square(sobel(gaussian_filtered_volume, axis = 0)) + np.square(sobel(gaussian_filtered_volume, axis = 1)) + np.square(sobel(gaussian_filtered_volume, axis = 2)))

    # enhancement w clahe
    # provides better results with using the gradient madnitude than without
    clahe_enhanced_volume = db2_clahe_enhance(gradient_mag)

    # apply morphological opening - spherical structure elements
    # helps with noise
    opened_volume = morphology.opening(clahe_enhanced_volume, morphology.ball(3)) # adjusted to work better w 3d volume

    return opened_volume

# grayscale to binary conversion -- not a global; adjustables determine region sizes
# works best since they have different illumination values 
# helps the clustering algo analyze the image better, without changing illumination differences between regions
def db2_thresholding(volume):
    block_size = 35
    db2_adaptive_thresholding = threshold_local(volume, block_size, offset=5)
    binary_adaptive = volume > db2_adaptive_thresholding
    return binary_adaptive

# perform dbscan
def dbscan_with_atlas(volume):

    # extracting textures features
    apply_texture_features = texture_features(volume)

    # extracting coordinates of non-zero voxels
    # i'm working on one where it extracts the background voxels instead
    db2_coords = np.column_stack(np.where(volume > 0))

    # getting the texture features of every non-zero voxel
    db2_iterate_texture_features = apply_texture_features[db2_coords[:, 0]]

    # combine texture features with the corresponding voxel coordinates 
    db2_combined_texture_features = np.hstack((db2_coords, db2_iterate_texture_features))

    # normalizing feature space
    db2_scaler = StandardScaler()
    db2_scaled_features = db2_scaler.fit_transform(db2_combined_texture_features)

    # run dbscan on scaled features
    db2 = DBSCAN(eps=1.5, min_samples=7).fit(db2_scaled_features)
    
    # Calculate cluster centers
    db2_cluster_centers = []
    labels = db2.labels_
    db2_unique_labels = np.unique(labels)
    for label in db2_unique_labels:
        if label != -1:  # Exclude noise label
            members = db2_coords[labels == label]
            center = members.mean(axis=0)
            db2_cluster_centers.append(center)

    db2_labeled_volume = "test"
    
    return db2, np.array(db2_cluster_centers), db2_labeled_volume

def db2_calculate_brightness(volume, db2_coords, labels):
    avg_brightness = {}
    db2_unique_labels = np.unique(labels)
    for label in db2_unique_labels:
        if label != -1:  # Exclude noise label
            members = db2_coords[labels == label]
            brightness_values = volume[members[:,0], members[:,1], members[:,2]]
            avg_brightness[label] = np.mean(brightness_values)
    return avg_brightness

# this would be called from core to execute all of the above
def db2_execute(volume, n_clusters):
    
    
    print("test")
    db2_preprocessed_volume = db2_preprocess(volume)
    print("test")
    db2_thresholded_volume = db2_thresholding(db2_preprocessed_volume)
    print("test")

    db2, db2_cluster_coords, db2_labeled_volume = dbscan_with_atlas(db2_thresholded_volume)
    print("test")
    avg_brightness = db2_calculate_brightness(volume, np.column_stack(np.where(db2_thresholded_volume > 0)), db2.labels_)
    print("test")
    
    return db2_cluster_coords, avg_brightness, db2_thresholded_volume

# returns output as a string
# i'm working on having the coordinates be in a similar format to the one you showed last night
# it's a pretty simple change, but i left it like this for now since this is what's been tested to work fully
def db2_output(db2_coordinates, avg_brightness):
    total_clusters = sum([len(db2_cluster_coords) for db2_cluster_coords in db2_coordinates])
    db2_results = f"Number of Clusters Found: {total_clusters}\n"
    
    db2_results += "Average Cluster Brightness:\n"
    for key, value in avg_brightness.items():
        db2_results += f"{key} : {value}\n"

    db2_results += "\n3D Coordinates of Clusters:\n"
    db2_cluster_count = 1
    for db2_volume_cluster_coords in db2_coordinates:
        for db2_cluster_coord in db2_volume_cluster_coords:
            db2_results += f"cluster {db2_cluster_count}: {db2_cluster_coord}\n"
            db2_cluster_count += 1

    return db2_results




## K-MEANS ##
# from core:
    # run "km_execute" to execute the algorithm
    # call "km_output" to display output

# identical to db2 preprocessing 
def km_preprocess(volume):
    
    gaussian_filtered_volume = gaussian_filter(volume, sigma=1)

    gradient_mag = np.sqrt(np.square(sobel(gaussian_filtered_volume, axis = 0)) + np.square(sobel(gaussian_filtered_volume, axis = 1)) + np.square(sobel(gaussian_filtered_volume, axis = 2)))

    clahe_enhanced_volume = clahe_enhance(gradient_mag)

    opened_volume = morphology.opening(clahe_enhanced_volume, morphology.ball(3))
    return opened_volume

# apply kmeans to preprocessed data
# initialize with specificied number of clusters to find
# maximum iterations are normally lower, but since we don't have a large dataset (in terms of kmeans), 1000 gave the most consistent outputs
def kmeans_clustering(volume, n_clusters=4, n_init='auto', max_iter=1000):
    apply_texture_features = texture_features(volume)

    # reshaping volume into a 2d array
    # each would be a different voxel
    reshaped_volume = volume.reshape(-1, 1)

    # run texture features on each voxel
    reshaped_features = np.repeat(apply_texture_features, volume.shape[1]*volume.shape[2], axis=0)
    
    # combines the intensity and voxel features
    km_combined_data = np.hstack([reshaped_volume, reshaped_features])

    # k-means++ is just the intialization method (set to auto so we dont need to actually change it each time we're running a new dataset)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=n_init, max_iter=max_iter)
    
    # fit model and combined data
    # predict cluster labels for each voxel
    labels = kmeans.fit_predict(km_combined_data)

    # matches labels to original volume shape, and returns them
    return labels.reshape(volume.shape), kmeans.cluster_centers_
#kmeans(and probably all the algos) are busted: they don't exclude zero-value clusters


# same brightness function, except this one's based on centers instead
def km_calculate_brightness(km_cluster_centers):
    avg_brightness = {}
    for label, center in enumerate(km_cluster_centers):
        avg_brightness[label] = center[0]
    return avg_brightness

# extracts 3d coordinates of voxels 
# applied to each cluster label in the volume
def km_extract_coordinates(km_labeled_volume):
    km_coordinates = {}

    # this is where it iterates over each unique label in the volume
    for label in np.unique(km_labeled_volume):
        km_coords = np.argwhere(km_labeled_volume == label)
        km_coordinates[label] = km_coords
    return km_coordinates

# this would be called from core to execute all of the above
def km_execute(volume, n_clusters):
    km_preprocessed_volume = km_preprocess(volume)
    km_labeled_volume, km_cluster_centers = kmeans_clustering(km_preprocessed_volume, n_clusters=n_clusters)
    km_coordinates = km_extract_coordinates(km_labeled_volume)
    avg_brightness = km_calculate_brightness(km_cluster_centers)
    return km_coordinates, km_labeled_volume, avg_brightness

# same set up as previous algo for getting the output info
def km_output(km_coordinates, avg_brightness):
    km_result = "Average Cluster Brightness:\n"
    for key, value in avg_brightness.items():
        km_result += f"{key} : {value}\n"

    km_result += "\n3D Coordinates of Clusters:\n"
    for key, value in km_coordinates.items():
        km_result += f"{key} : {value}\n"

    return km_result


## Hierarchical ##
# from core:
    # run "hr_execute" to execute the algorithm
    # call "hr_output" to display output

# anything uncommented/unexplained will have an identical function/process in the above algos
# it'll have any necessary explanation there

def hr_preprocess(volume):

    gaussian_filtered_volume = apply_gaussian_filter(volume)
    
    gradient_mag = np.sqrt(np.square(sobel(gaussian_filtered_volume, axis=0)) + np.square(sobel(gaussian_filtered_volume, axis=1)) + np.square(sobel(gaussian_filtered_volume, axis=2)))

    clahe_enhanced_volume = clahe_enhance(gradient_mag)

    opened_volume = morphology.opening(clahe_enhanced_volume, morphology.ball(3))
    return opened_volume

def hr_clustering(volume, n_clusters=4):
    # extract features
    apply_texture_features = texture_features(volume)
    reshaped_volume = volume.reshape(-1, 1)
    reshaped_features = np.repeat(apply_texture_features, volume.shape[1]*volume.shape[2], axis=0)
    hr_combined_data = np.hstack([reshaped_volume, reshaped_features])

    # perform hierarchical clustering
    # can do the euclidean affinity matrix and ward linkage through the same agglomerative clustering submodule
    # this helps a bit with the memory issue since we're not runnign them separately anymore
    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward') # ward minimizes the "within-cluster" variance
    labels = clustering.fit_predict(hr_combined_data)
    return labels.reshape(volume.shape)

# although it's the same function as above, it's not the best yet since i'm still testing which hr cluster features would be best to call here
def hr_calculate_brightness(hr_labeled_volume):
    avg_brightness = {}
    unique_labels = np.unique(hr_labeled_volume)
    for label in unique_labels:
        avg_brightness[label] = np.mean(hr_labeled_volume[hr_labeled_volume == label])
    return avg_brightness

def hr_extract_coordinates(hr_labeled_volume):
    hr_coordinates = {}
    for label in np.unique(hr_labeled_volume):
        hr_coords = np.argwhere(hr_labeled_volume == label)
        hr_coordinates[label] = hr_coords
    return hr_coordinates

def hr_execute(volume):
    hr_preprocessed_volume = hr_preprocess(volume)
    hr_labeled_volume = hr_clustering(hr_preprocessed_volume)
    hr_coordinates = hr_extract_coordinates(hr_labeled_volume)
    avg_brightness = hr_calculate_brightness(hr_labeled_volume)
    return hr_coordinates, hr_labeled_volume, avg_brightness

def hr_output(hr_coordinates, avg_brightness):
    hr_result = "Average Cluster Brightness:\n"
    for key, value in avg_brightness.items():
        hr_result += f"{key} : {value}\n"

    hr_result += "\n3D Coordinates of Clusters:\n"
    for key, value in hr_coordinates.items():
        hr_result += f"{key} : {value}\n"

    return hr_result


#hardcoded, all of clustering is going to be overwritten by MD anyway
'''
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
'''


def convert_to_lists(dict_of_arrays):
    dict_of_lists = {}
    for key, array in dict_of_arrays.items():
        # Swap the first and third columns (axis)
        array[:, [0, 2]] = array[:, [2, 0]]

        # Convert each 2D array into a list of lists
        list_of_lists = array.tolist()
        dict_of_lists[key] = list_of_lists
    return dict_of_lists

#moved to bottom, just to keep sep from everything else
def execute_whole_clustering(input, algo, n_clusters):
    """
    input: an entire scan (3d np array) and a string representing the chosen algo
    selects the algo from a dictionary of corresponding functions
    output: a dictionary of region : voxel coordinate lists
    """
     # initialize dictionary to store output
    output_coords = {}

    #dict of strings that correspond to functions
    algos_dict = {
        'DBSCAN': db2_execute,
        'K-Means': km_execute,
        'Hierarchical': hr_execute
    } 

    output_coords, labeled_volume, avg_brightness = algos_dict[algo](input, n_clusters)
    
    output_coords = convert_to_lists(output_coords)
    
    return output_coords

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

def tester_algo(input_array):
    """
    input: np array
    prints
    output: format for the output of any clustering algo
    """
    print("clustering testing algo")
    test_coords = [[x, y, z] for x in range(0, 30) for y in range(0, 30) for z in range(0, 30)]
    return test_coords


def execute_seg_clustering(input, algo, n_clusters):
    """
    input: an pre-atlas segmented scan (dict of 3d np arrays) and a string representing the chosen algo
    selects the algo from a dictionary of corresponding functions
    output: a dictionary of region : voxel coordinate lists
    """
    # initialize dictionary to store output
    output_coords_dict = {}

    #dict of strings that correspond to functions
    algos_dict = {
        'DBSCAN': db2_execute,
        'K-Means': km_execute,
        'Hierarchical': hr_execute
    }
    
    for region, scan in input.items():
        output_coords, labeled_volume, avg_brightness = algos_dict[algo](scan, n_clusters)
        output_coords = convert_to_lists(output_coords)
        output_coords_dict[region] = output_coords

    
        
    return output_coords_dict



# used as main sript, this helps a lot with testing and pinpointing errors.
#I'm already working on creating function shortcuts and combining factors for easy use as a sub-module instead.
if __name__ == "__main__":
    #folder_path = input("Enter folder path: ") # get folder
    folder_path = "scan1"
    volume = data.get_3d_image(folder_path) # create 3d volume

    coordinates = execute_whole_clustering(volume, "DBSCAN", 2)

    clustered_dict = segmentation.create_seg_images_from_image(volume, coordinates)
    data.display_seg_np_images(clustered_dict)

    # apply dbscan to 3d and get labels, overall coordinates, and binary masks
    #labeled_volume, cluster_coords, brain_mask, skull_mask = dbscan_3d(volume)

    # find brain and skull coordinates
    #brain_cluster_coordinates, skull_cluster_coordinates = cluster_coordinates(cluster_coords, brain_mask, skull_mask)

    #display_slices(volume, labeled_volume, cluster_coords, brain_mask, skull_mask)

    # print("3D Brain Cluster Coordinates:")
    # print(brain_cluster_coordinates)

    # print("3D Skull Cluster Coordinates:")
    # print(skull_cluster_coordinates) 

