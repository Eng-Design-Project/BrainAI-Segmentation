import segmentation
import data
import numpy as np
import matplotlib.pyplot as plt


# image = data.get_3d_image("scan1")

# # use functions in data to read the atlas and 
# #   the image in question into memory here as 3d np array images


# def avg_bright_arr(images):
#     # Ensure the input is a numpy array
#     if not isinstance(images, np.ndarray):
#         raise ValueError("Input must be a numpy array")
#     # Ensure the input is a 3D array
#     if len(images.shape) != 3:
#         raise ValueError("Input must be a 3D array of grayscale images")
#     # Calculate the average pixel brightness for all images
#     average_brightness = np.mean(images, axis=(1, 2))
#     return average_brightness

# image_depth = image.shape[0]   
# atlas_paths = data.get_atlas_path(image_depth)
# #if atlas_paths = 0  popup "Image too large" break out of function
# atlas = data.get_3d_image(atlas_paths[0])
# # get atlas colors as 3d np array
# color_atlas = data.get_2d_png_array_list(atlas_paths[1])

# seg_results = segmentation.execute_atlas_seg(atlas, color_atlas, image)
# #print("shape:")
# #print(seg_results['Brain']) #this returns a 3d array of images



# def array_of_average_pixel_brightness_3d(images):
#     # Ensure the input is a numpy array
#     if not isinstance(images, np.ndarray):
#         raise ValueError("Input must be a numpy array")
#     # Ensure the input is a 3D array
#     if len(images.shape) != 3:
#         raise ValueError("Input must be a 3D array of grayscale images")

#     num_slices, height, width = images.shape

#     # Initialize an array to store the normalized average brightness for each image
#     average_brightness = np.zeros(num_slices)

#     for i in range(num_slices):
#         # Check if the range is zero to avoid division by zero
#         if np.max(images[i]) - np.min(images[i]) == 0:
#             normalized_image = images[i]  # Avoid normalization if the range is zero
#         else:
#             # Normalize each image to the range [0, 255]
#             normalized_image = (images[i] - np.min(images[i])) / (np.max(images[i]) - np.min(images[i])) * 255

#         # Calculate the average pixel brightness for the normalized image
#         average_brightness[i] = np.mean(normalized_image)

#     return average_brightness

# avg_bright_array = array_of_average_pixel_brightness_3d(image)
# images = seg_results['Brain']



# print("single slice:")
# slice1 = image[0]
# print(slice1.shape)
# print(np.min(slice1))
# print(np.max(slice1))

# mean = np.mean(slice1)
# avg = data.avg_brightness_2d(slice1)

# print("average (not normalized) brightness of first slice of scan1:")
# print(mean)
# print("average (normalized) brightness of first slice of scan1:")
# print(avg)

# print("average (not normalized) divided by max")
# print(mean/np.max(slice1))
# print("average (normalized) divided by 255")
# print(avg/255)
# print("if the 2 ratios are the same, then the normalization is correct")

# plt.imshow(seg_results['Brain'][5], 'gray', origin='upper')
# plt.show()

# print("first avg of the normalized averages array:")
# print(avg_bright_array[0])

# print(data.array_of_average_pixel_brightness_3d(seg_results['Brain']))

a1=[1,2,3,4,5,6]  
b1=[[1,2,3], [4,5,6]]
ar1 = np.array(a1)
ar2 = np.array(b1)
print(ar1.shape)
print(ar2.shape)


