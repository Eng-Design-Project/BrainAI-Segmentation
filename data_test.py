#use get 3d image (sitk)
#use the numpy version
# testing the following functions from data: view_np_3d_image(), display_seg_np_images(), and convert_sitk_dict_to_numpy()

import data
import SimpleITK as sitk
import os
import numpy as np

# first argument should be a higher level folder with brain region subfolders containing DCM files.
# the output is a dictionary with brain region names as keys and np arrays(images) as values
# note: this function isn't useful because I run into problems when coverting segmented DCMs to numpy arrays
def subfolders_to_np_dictionary(directory):
    #print(os.listdir(directory))
    region_dict = {}
    for i in os.listdir(directory):
        # print(i)
        region_dict[i] = data.get_3d_image(os.path.join(directory, i))

    return region_dict


#Testing functions which display sitk images
# image1 = data.get_3d_image("scan1")
# image1 = sitk.GetArrayFromImage(image1)
# #data.display_3d_array_slices(image1, 10)

# #Testing functions to display numpy 3d images (works)
# image2 = data.get_3d_image("scan1")
# #data.display_3d_array_slices(image2, 10)


# # now I want to get np dict, display it
# np_dict = data.subfolders_to_dictionary("atl_segmentation_DCMs")
# # this should display 5 slices
# data.display_seg_np_images(np_dict)


# # statement below doesn't work, I presume because the segmented DCMs lack the relevant metadata (we use pydicom.dcmread() to convert DCMs into np arrays)
#     #"subfolders_to_np_dictionary" uses "get_3d_array_from_file" which is causing the error
# # np_dict = subfolders_to_np_dictionary("atl_segmentation_DCMs")
# # display_seg_np_images(np_dict)

# #array = sitk.GetArrayFromImage(sitk_image) is used to convert sitk image to np array
# # I want a function to convert a dict of regions:sitk images to a dict of regions:np_arrays
# # I should find a function that takes an sitk image and turns it into a dictionary

# #data.view_np_3d_image(np_dict["Brain"],10, "brain")


# # three_d_arr = data.get_3d_array_from_file("scan1")
# # arr = three_d_arr[:][:][44] #this shold be the numpy array
# # arr = np.interp(arr, (arr.min(), arr.max()), (0, 255))
# # arr = np.uint8(arr)
# # png = Image.fromarray(arr)
# # plt.imsave("a_test.png", png, cmap="gray")

# Create a sample 3D numpy array with two grayscale images
image1 = np.array([[10, 20, 30],
                      [40, 50, 60],
                      [70, 80, 90]])

image2 = np.array([[100, 110, 120],
                      [130, 140, 150],
                      [160, 170, 180]],)

image3 = np.array([[100, 110, 120],
                      [130, 140, 150],
                      [160, 170, 180]],)

grayscale_images = np.array([image1, image2, image3])

# Calculate the average pixel brightness for all images
avg_brightness = data.array_of_average_pixel_brightness_3d(grayscale_images)
print(f"Average Pixel Brightness for Each Image: {avg_brightness}")

scan1 = data.get_3d_image("scan1") #output should be a numpy 3d array
# Calculate the average pixel brightness for all images
avg_bright = data.array_of_average_pixel_brightness_3d(scan1)
print(f"Average Pixel Brightness for 3d Image: {avg_bright}")
print("type: ", type(avg_bright))
print(avg_bright[0])
print(scan1.shape)
print(scan1[0].shape)

def average_brightness(image):
    # Ensure the input image is a numpy array
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a numpy array")

    # Ensure the image is 2D (grayscale)
    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale (2D array)")

    # Calculate the average pixel brightness
    average_brightness = np.mean(image)
    return average_brightness

av = average_brightness(scan1[0])
print(av)
print(image1.shape)

avgg = data.average_overall_brightness_3d(scan1)
print(avgg)