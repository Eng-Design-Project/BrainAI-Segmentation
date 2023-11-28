import segmentation
import data
import numpy as np

image = data.get_3d_image("scan1")

# use functions in data to read the atlas and 
#   the image in question into memory here as 3d np array images


def avg_bright_arr(images):
    # Ensure the input is a numpy array
    if not isinstance(images, np.ndarray):
        raise ValueError("Input must be a numpy array")
    # Ensure the input is a 3D array
    if len(images.shape) != 3:
        raise ValueError("Input must be a 3D array of grayscale images")
    # Calculate the average pixel brightness for all images
    average_brightness = np.mean(images, axis=(1, 2))
    return average_brightness

image_depth = image.shape[0]   
atlas_paths = data.get_atlas_path(image_depth)
#if atlas_paths = 0  popup "Image too large" break out of function
atlas = data.get_3d_image(atlas_paths[0])
# get atlas colors as 3d np array
color_atlas = data.get_2d_png_array_list(atlas_paths[1])

seg_results = segmentation.execute_atlas_seg(atlas, color_atlas, image)
#print("shape:")
#print(seg_results['Brain']) #this returns a 3d array of images



def array_of_average_pixel_brightness_3d(images):
    # Ensure the input is a numpy array
    if not isinstance(images, np.ndarray):
        raise ValueError("Input must be a numpy array")
    # Ensure the input is a 3D array
    if len(images.shape) != 3:
        raise ValueError("Input must be a 3D array of grayscale images")

    num_slices, height, width = images.shape

    # Initialize an array to store the normalized average brightness for each image
    average_brightness = np.zeros(num_slices)

    for i in range(num_slices):
        # Check if the range is zero to avoid division by zero
        if np.max(images[i]) - np.min(images[i]) == 0:
            normalized_image = images[i]  # Avoid normalization if the range is zero
        else:
            # Normalize each image to the range [0, 255]
            normalized_image = (images[i] - np.min(images[i])) / (np.max(images[i]) - np.min(images[i])) * 255

        # Calculate the average pixel brightness for the normalized image
        average_brightness[i] = np.mean(normalized_image)

    return average_brightness

avg_bright_array = array_of_average_pixel_brightness_3d(seg_results['Brain'])
print(avg_bright_array)
print('......')
images = seg_results['Brain']
print( np.min(images), np.max(images), np.min(images)*255)
print(avg_bright_arr(seg_results['Brain']))
