#dicom_path2 = "ADNI_007_S_1339_PT_TRANSAXIAL_BRAIN_3D_FDG_ADNI_CTAC__br_raw_20070402142342176_8_S29177_I47688.dcm"
#dicom_path1 = "ADNI_007_S_1339_PT_TRANSAXIAL_BRAIN_3D_FDG_ADNI_CTAC__br_raw_20070402142342697_9_S29177_I47688.dcm"

import numpy as np
import matplotlib.pyplot as plt
import data

import numpy as np
from scipy.ndimage import affine_transform
from scipy.signal import fftconvolve

def create_black_copy(input_array: np.ndarray) -> np.ndarray:
    # Create a new array with the same shape and type as the input array, with all elements set to zero
    black_np_array = np.zeros_like(input_array)
    return black_np_array


#for expand region of interest
from scipy.ndimage import convolve

#deprecated, no longer using sitk - Dustin
# def array_to_image_with_ref(data: np.ndarray, reference_image: sitk.Image) -> sitk.Image:
#     # Convert the numpy array to a SimpleITK image
#     new_image = sitk.GetImageFromArray(data)
    
#     # Set the spatial information from the reference_image
#     new_image.SetSpacing(reference_image.GetSpacing())
#     new_image.SetOrigin(reference_image.GetOrigin())
#     new_image.SetDirection(reference_image.GetDirection())

#     return new_image

#this is the currently used
def scipy_register_images(target, moving):
    """
    Register the moving image to the target image using cross-correlation and return the transformed image.
    """
    
    # Calculate the cross-correlation in 3D
    cross_correlation = fftconvolve(target, moving[::-1, ::-1, ::-1], mode='same')
    
    # Find the shift for which cross-correlation is maximum
    shifts = np.unravel_index(np.argmax(cross_correlation), cross_correlation.shape)

    # Calculate how much to shift to align the images at the center
    center_shifts = np.array(target.shape) / 2 - np.array(shifts)
    
    # Use affine transform to shift the moving image
    registered_image = affine_transform(moving, matrix=np.eye(3), offset=center_shifts)
    
    return registered_image

def test_scipy_register_images(atlas, image):
    print("testing scipy image reg")
    #get 3d np array images
    moving_image = data.get_3d_image(image)
    target_image = data.get_3d_image(atlas)
   
    
    #register the 3d array to atlas 3d array
    reg_image = scipy_register_images(target_image, moving_image)
    
    #save registered image as dcm first, then as png
    data.save_3d_img_to_dcm(reg_image, "scipy_reg_image_dcm")
    data.save_dcm_dir_to_png_dir("scipy_reg_image_dcm", "scipy_reg_png")

    
#test_scipy_register_images("atlas", "scan2")

# Example usage
#target_image = np.random.rand(100, 100, 100)
#moving_image = np.roll(target_image, shift=5, axis=0)  # Let's shift target image by 5 pixels in x-direction
#registered_image = scipy_register_images(target_image, moving_image)
# Now, the 'registered_image' should be closely aligned with 'target_image'


#expand region of interest
#this adds an extra layer of pixels to a segmented image from the original image
#takes 3d np array images now
def expand_roi(original_arr, segment_arr, layers=5):
    """
    Expand the region of interest in the segment_arr based on the original_arr.
    
    :param original_arr: The original array.
    :param segment_arr: The array representing the segment to be expanded.
    :param layers: Number of layers to expand.
    :return: The expanded segment array.
    """

    # Define a kernel for 3D convolution that checks for 26 neighbors in 3D
    kernel = np.ones((3, 3, 3))
    kernel[1, 1, 1] = 0  # We don't want the center pixel

    # Initialize the expanded segment array with the initial segment
    expanded_segment_arr = segment_arr.copy()

    for _ in range(layers):
        # Convolve with the segment to find the boundary of ROI
        boundary = convolve(expanded_segment_arr > 0, kernel) > 0
        boundary[expanded_segment_arr > 0] = 0  # Remove areas that are already part of the segment

        # Copy pixel values from the original image to the boundary in the expanded segment
        expanded_segment_arr[boundary] = original_arr[boundary]

    return expanded_segment_arr


# Example usage:
# original = np.random.rand(10, 10, 10)
# segment = np.zeros((10, 10, 10))
# segment[4:7, 4:7, 4:7] = 1
# result = expand_roi(original, segment)

#NOT CURRENTLY USED, SITK registration never worked
# def atlas_segment(atlas, image, 
#                   simMetric="MeanSquares", optimizer="GradientDescent", 
#                   interpolator="Linear", samplerInterpolator="Linear"):

#     #set up the registration framework
#     registration_method = sitk.ImageRegistrationMethod()

#     #set similarity metric
#     if simMetric == "MeanSquares":
#         registration_method.SetMetricAsMeanSquares()
#     else:
#         print("default sim metric: MeanSquares")
#         registration_method.SetMetricAsMeanSquares()

#     #set optimizer
#     if optimizer == "GradientDescent":
#         registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
#         registration_method.SetOptimizerScalesFromPhysicalShift()
#     else:
#         print("default optimizer: GradientDescent")
#         registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
#         registration_method.SetOptimizerScalesFromPhysicalShift()

#     #initial transform
#     #initial_transform = sitk.TranslationTransform(atlas.GetDimension())
#         #transforms only translation? affline instead?
#     #Rigid Transform (Rotation + Translation):
#     #initial_transform = sitk.Euler3DTransform()
#     #Similarity Transform (Rigid + isotropic scaling):
#     #initial_transform = sitk.Similarity3DTransform()
#     #Affine Transform (includes rotations, translations, scaling, and shearing):
#     initial_transform = sitk.AffineTransform(atlas.GetDimension())
#     #BSpline Transform (a non-rigid, deformable transform): *DOES NOT CURRENTLY WORK*
#     #order_x, order_y, order_z = 5, 5, 5
#     #initial_transform = sitk.BSplineTransformInitializer(atlas, [order_x, order_y, order_z])


#     registration_method.SetInitialTransform(initial_transform)

#     #set interpolator
#     if interpolator == "Linear":
#         registration_method.SetInterpolator(sitk.sitkLinear)
#     else:
#         print("default interpolator: Linear")
#         registration_method.SetInterpolator(sitk.sitkLinear)

#     #execute registration
#     final_transform = registration_method.Execute(sitk.Cast(atlas, sitk.sitkFloat32), sitk.Cast(image, sitk.sitkFloat32))

#     #apply transformation
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetReferenceImage(atlas)
#     if samplerInterpolator == "Linear":
#         resampler.SetInterpolator(sitk.sitkLinear)
#     elif samplerInterpolator == "HigherOrder":
#         resampler.SetInterpolator(sitk.sitkBSpline)
#     elif samplerInterpolator == "NearestNeighbor":
#         resampler.SetInterpolator(sitk.sitkNearestNeighbor)
#     else:
#         print("default samplerInterpolator Linear")
#         resampler.SetInterpolator(sitk.sitkLinear)
    
#     resampler.SetDefaultPixelValue(100)
#     resampler.SetTransform(final_transform)

#     registered_image = resampler.Execute(image)
#     return registered_image

#NOT USED, SITK registration never worked
# def test_atlas_segment_hardcoded():
#     # Path to the directory that contains the DICOM files
#     atlas_dir = "scan1"
#     input_dir = "scan2"
#     # Create 3d image with SITK
#     atlas_image = data.get_3d_image(atlas_dir)
#     input_image = data.get_3d_image(input_dir)
#     #does it need to by cast to float32?

#     study_id = input_image.GetMetaData('0020|000D') if input_image.HasMetaDataKey('0020|000D') else ""
#     series_id = input_image.GetMetaData('0020|000E') if input_image.HasMetaDataKey('0020|000E') else ""
#     print("study id: ", study_id)
#     print("series id: ", series_id)

#     registered_image = atlas_segment(atlas_image, input_image)

#     #data.view_sitk_3d_image(map_image, 5, "map image")
#     #data.view_sitk_3d_image(input_image, 5, "input image")
#     #data.view_sitk_3d_image(registered_image, 5, "registered image")

#     data.save_3d_img_to_dcm(registered_image, "registered")
#     data.save_dcm_dir_to_png_dir("registered", "reg pngs")
#test_atlas_segment_hardcoded()

#data.save_dcm_dir_to_png_dir("atlas", "atlas pngs")
#data.save_dcm_dir_to_png_dir("registered", "reg pngs")

#THIS FUNCTION WILL BE DEPRECATED SOON, AS THERE IS A FUNCTION IN THE DATA MODULE THAT DOES IT MORE SIMPLY
#given a dictionary with region names as keys and sitk images as values, this funciton displays them
# def display_regions_from_dict(region_images):
#     for region_name, region_image in region_images.items():
#         print(region_name)
#         print(region_image.GetSize())

#         plt.figure(figsize=(6, 6))
#         array_from_image = sitk.GetArrayFromImage(region_image)
#             # Displaying the first slice of the 3D image
#         plt.imshow(array_from_image[0, :, :], cmap='gray')
#         plt.axis('off')
#         plt.title(f"Region: {region_name}")
#         plt.show()

def create_seg_images_from_image(image, region_dict):
    output_images = {}
    for region_name, coordinates_list in region_dict.items():
        blank_image = create_black_copy(image)
        
        for coordinates in coordinates_list:
            x, y, z = coordinates
            if (0 <= z < image.shape[0]) and \
               (0 <= y < image.shape[1]) and \
               (0 <= x < image.shape[2]):
                pixel_value = image[z, y, x]
                blank_image[z, y, x] = pixel_value
                
        # Append the finished blank_image to the output_images dictionary
        output_images[region_name] = blank_image

    print(f"Size of output images:  {len(output_images)}")

    return output_images

def filter_noise_from_images(images_dict, noise_coords_dict):
    # Ensure the noise coordinates dictionary has keys that exist in the images dictionary
    if not set(noise_coords_dict.keys()).issubset(set(images_dict.keys())):
        raise ValueError("Keys in noise_coords_dict should be a subset of images_dict.")
    
    # Create a copy of the images dictionary to modify and return
    filtered_images = {k: np.copy(v) for k, v in images_dict.items()}

    # For each brain region in the noise_coords_dict
    for region, coords_list in noise_coords_dict.items():
        # Get the 3D image for the region
        
        # Iterate over the coordinates in coords_list
        for x, y, z in coords_list:
            # Ensure the coordinates are within the image's bounds
            if (0 <= x < filtered_images[region].shape[0]) and \
               (0 <= y < filtered_images[region].shape[1]) and \
               (0 <= z < filtered_images[region].shape[2]):
                filtered_images[region][x, y, z] = 0  # Set the voxel to black
    
    return filtered_images

#IMPORTANT: #This function assumes the given coordinates are the ones we want to keep.
# more efficient to filter out coordinates we dont want, use the filter noise function instead
#this would take a dict of atlas segmented images, and then further refine them with coordinates output by an 
# Advanced Segmentation algo, with corresponding region names
def create_seg_images_from_dict(images_dict, coords_dict):
    output_images = {}

    for region_name, coordinates_list in coords_dict.items():
        # Ensure that the region_name exists in the images_dict
        if region_name not in images_dict:
            print(f"Warning: No image found for region {region_name}")
            continue

        current_image = images_dict[region_name]
        blank_image = create_black_copy(current_image)

        for coordinates in coordinates_list:
            x, y, z = coordinates
            if (0 <= x < current_image.shape[0]) and \
               (0 <= y < current_image.shape[1]) and \
               (0 <= z < current_image.shape[2]):
                pixel_value = current_image[z, y, x]
                blank_image[z, y, x] = pixel_value

        # Append the finished blank_image to the output_images dictionary
        output_images[region_name] = blank_image

    print(f"Size of output images:  {len(output_images)}")

    return output_images

# Deprecated: SITK, and was just a tester function
# takes a directory of DCMs, outputs a dictionary with region names as keys and sitk images as the values
# def DCMs_to_sitk_img_dict(directory):
#     image = data.get_3d_image(directory)
"""
#     #this part of the function could be expanded to have more regions
#     def generate_regions(): 
#         region1 = [[x, y, z] for x in range(0, 51) for y in range(0, 51) for z in range(0, 51)]
#         region2 = [[x, y, z] for x in range(50, 101) for y in range(50, 101) for z in range(0, 50)]

#         region_dict = {
#             "Region1": region1,
#             "Region2": region2
#         }
#         return region_dict
    
#     # Define your regions and their coordinates here
#     region_dict = generate_regions()
#     region_images = create_seg_images_from_image(image, region_dict)
#     #display_regions_from_dict(region_images)
#     data.display_seg_np_images(region_images)
    """
#DCMs_to_sitk_img_dict("scan1")

#extra pixel layer algo
#takes the registered scan, a brain region scan
#for all non-black voxels, note coordinates of adjacent black voxels
#copy pixel data at those coordinates from the registered image 
# to a copy of brain region image
#return 'blurred' brain region image


#registration testing
#get two similar images
#register one to the other
#display both original and new for comparison

#optimizers
#gradient descent-based methods and LBFGS 
# Gradient Descent Optimizer and its variants like 
# Gradient Descent Line Search, Regular Step Gradient Descent,
#  and Conjugate Gradient are often used.

#similarity metrics
#start with mean squares or normalized cross-correlation. 
# If these do not give satisfactory results, you may try mutual information.

#interpolator
#sitk.sitkLinear

#takes 3d array, new version takes list of 2d arrays
# def encode_atlas_colors(image_3d: np.ndarray) -> dict:
#     #hard coded colors to region
#     color_to_region_dict = {
#         (237, 28, 36): 'Skull',   # Redish
#         (0, 162, 232): 'Brain',   # Blueish
#         #... add other colors and regions as required
#     }
#     # Initialize the output dictionary with region names as keys and empty lists as values
"""#     region_coords_dict = {region: [] for region in color_to_region_dict.values()}

#     # Iterate through the 3D array to get the coordinates and pixel values
#     for x in range(image_3d.shape[0]):
#         for y in range(image_3d.shape[1]):
#             for z in range(image_3d.shape[2]):
#                 #pixel_color = tuple(image_3d[x, y, z])#note, xyz from 3d arrays
#                 pixel_color = tuple(image_3d[x, y, :, z])
#                 if pixel_color != (0, 0, 0):
#                     print(pixel_color)
#                 # If the pixel color exists in the dictionary, add its coordinate to the respective list
#                 if pixel_color in color_to_region_dict:
#                     region = color_to_region_dict[pixel_color]
#                     region_coords_dict[region].append([z, y, x])#note zyx for comparison to sitk images
#                     print(region, + ": ", + pixel_color)
#     return region_coords_dict"""


def encode_atlas_colors(image_list: list, color_to_region_dict: dict) -> dict:
    """
    Create a dictionary mapping regions to their corresponding coordinates based on exact color matches.

    Args:
    - image_list: List of 2D numpy arrays representing slices of the 3D image.
    - color_to_region_dict: Dictionary mapping exact colors to region names.

    Returns:
    - Dictionary mapping region names to lists of coordinates where those regions are found.
    """
    # Initialize the output dictionary with region names as keys and empty lists as values
    region_coords_dict = {region: [] for region in color_to_region_dict.values()}

    # Iterate through the list of 2D arrays to get the coordinates and pixel values
    for z, image_2d in enumerate(image_list):
        for y in range(image_2d.shape[0]):
            for x in range(image_2d.shape[1]):
                # Get the pixel color as a tuple
                pixel_color = tuple(image_2d[y, x])

                # If the pixel color exactly exists in the dictionary, add its coordinate to the respective list
                if pixel_color in color_to_region_dict:
                    region = color_to_region_dict[pixel_color]
                    region_coords_dict[region].append([x, y, z])

    return region_coords_dict


def test_encode_atlas_colors():
    # Define image dimensions
    width = 40
    height = 40
    depth = 40
    # Create an empty RGB 3D image
    image_3d = np.zeros((depth, height, width, 3), dtype=np.uint8)
    # Fill the left with red
    image_3d[:, :, :width//3] = [237, 28, 36]
    # Fill the middle with blue
    image_3d[:, :, width//3:2*width//3] = [0, 162, 232]
    # Fill the right with green
    image_3d[:, :, 2*width//3:] = [0, 255, 0]
    #get a dict of regions and coords
    color_to_region_dict = {
        (237, 28, 36): 'Skull',   # Redish
        (0, 162, 232): 'Brain',   # Blueish
        # ... add other colors and regions as required
    }
    region_to_coord_dict = encode_atlas_colors(image_3d, color_to_region_dict)
    #print(region_to_coord_dict)
    data.view_np_3d_image(image_3d, 5, "redbluegreen")

    #create image dict from coords dict and image_3d
    final_dict = create_seg_images_from_image(image_3d, region_to_coord_dict)

    #test expand_roi()
    
    #following is code to test expand ROI, but it doesn't work on RGB images
    # this is fine, we only need it to work on grayscale
    # for region, image in final_dict.items():
    #     for x in range(10):
    #         #RuntimeError: filter weights array has incorrect shape.
    #         image_3d = expand_roi(np_original, image_3d)
    #data.display_seg_np_images(final_dict)
    print(image_3d.shape)


#atlas and image are both 3d np arrays, atlas colors is a list of 2d np arrays
def execute_atlas_seg(atlas, atlas_colors, image):
    print("executing atlas seg")
   
    #register the 3d array to atlas 3d array
    reg_image = scipy_register_images(atlas, image)
    
    #coordinates of region based on atlas
    color_to_region_dict = {
        (237, 28, 36): 'Skull',   # Redish
        (236, 28, 36): 'Skull',   # Redish
        (0, 162, 232): 'Brain',   # Blueish
        (0, 168, 243): 'Brain',   # Blueish
        # ... add other colors and regions as required
    }
    region_to_coord_dict = encode_atlas_colors(atlas_colors, color_to_region_dict)

    #create image dict from coords dict and 3d array image
    final_dict = create_seg_images_from_image(reg_image, region_to_coord_dict)

    #expand roi
    for region, segment in final_dict.items():
        final_dict[region] = expand_roi(reg_image, segment)

    #return final_dict
    #return region_to_coord_dict and final dict (it'll return a tuple)
    return final_dict, region_to_coord_dict

#can only be done after normal atlas seg
def execute_internal_atlas_seg(image_dict: dict, internal_color_atlas: list) -> dict:
    print("executing internal atlas seg")
    internal_dict = {}
    for region in image_dict.keys():
        if region == "Brain":
            print("Segmenting Brain")
            color_to_region_dict = {
                (236, 28, 36): 'White Matter',   # Red
                (184, 61, 186): 'Frontal',   # Pink
                (63, 72, 204): 'Temporal',   # Blue
                (185, 122, 86): 'Occipital',   # Brown
                # ... add other colors and regions as required
            }

            internal_color_atlas_coords = encode_atlas_colors(internal_color_atlas, color_to_region_dict)
            internal_dict = create_seg_images_from_image(image_dict[region], internal_color_atlas_coords)
            for internal_region, segment in internal_dict.items():
                internal_dict[internal_region] = expand_roi(image_dict[region], segment)

    #return internal_dict
    return internal_dict, internal_color_atlas_coords





if __name__ == "__main__":
    atlas_path = data.get_atlas_path(45)
    atlas = data.get_3d_image(atlas_path)
    image = data.get_3d_image("scan1")
    #data.display_3d_array_slices(image, 10)
    
    color_atlas = data.get_2d_png_array_list("color atlas")
    seg_results = execute_atlas_seg(atlas, color_atlas, image)
    
    internal_color_atlas = data.get_2d_png_array_list("Color Atlas internal")
    internal_seg_results = execute_internal_atlas_seg(seg_results, internal_color_atlas)

    data.display_seg_np_images(internal_seg_results)

    #test_encode_atlas_colors()

    #data.display_seg_np_images(seg_results)

    #data.store_seg_img_on_file(seg_results, "scan1", "dustin atlas tes one million")
    #data.store_seg_png_on_file(seg_results,"dustin atlas tes one million pngs")

    # coords_dict = {
    # "Brain": [(x, y, z) for x in range(30) for y in range(128) for z in range(46)]
    # }
    #seg_dict_from_seg_dict = create_seg_images_from_dict(seg_results, coords_dict)
    #data.display_seg_np_images(seg_dict_from_seg_dict)


'''
# Replace 'image.dcm' with the path to your DICOM file
image = sitk.ReadImage('mytest.dcm')

pixarray = sitk.GetArrayFromImage(image)
pixarray = np.squeeze(pixarray)
print(pixarray.ndim)
plt.imshow(pixarray, 'gray', origin='lower')
plt.show()

# Print the size of the image
print(image.GetSize())

# def segment_images(*images):
#this will recieve a tuple of unspecified size as arguments, 
#a loop will go through and call segment_image on each



def segment_image(image):
    #register image to unmarked atlas
    #create a blank image (3d np array)
    #each region has a set of 3d coordinates, created from the atlas
    #for each region, at the atlas coordinates
    #   each entry is a tuple of two values:
    #   the pixel value of the registered image at the same coordinate
    #   a number to indicate the region of that pixel

    print("function incomplete")


    #some known sitk methods
    
    #sitk image copied into numpy array
    npy_array = sitk.GetArrayFromImage(image)

    #np array to sitk image
    new_image = sitk.GetImageFromArray(npy_array)

    #return a 3d array of tuples: each tuple is the pixel value of the registered image 
    # and a number indicating the region



  
#registration testing
#get two similar images
#register one to the other
#display both original and new for comparison

#optimizers
#gradient descent-based methods and LBFGS 
# Gradient Descent Optimizer and its variants like 
# Gradient Descent Line Search, Regular Step Gradient Descent,
#  and Conjugate Gradient are often used.

#similarity metrics
#start with mean squares or normalized cross-correlation. 
# If these do not give satisfactory results, you may try mutual information.

#interpolator
#sitk.sitkLinear
'''