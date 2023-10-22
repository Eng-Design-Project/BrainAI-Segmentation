#dicom_path2 = "ADNI_007_S_1339_PT_TRANSAXIAL_BRAIN_3D_FDG_ADNI_CTAC__br_raw_20070402142342176_8_S29177_I47688.dcm"
#dicom_path1 = "ADNI_007_S_1339_PT_TRANSAXIAL_BRAIN_3D_FDG_ADNI_CTAC__br_raw_20070402142342697_9_S29177_I47688.dcm"

import numpy as np
import matplotlib.pyplot as plt
#import SimpleITK as sitk
import data
import pydicom
from scipy.ndimage import affine_transform
from scipy.signal import fftconvolve
#for expand region of interest
from scipy.ndimage import convolve
from pydicom.uid import ExplicitVRLittleEndian 
from scipy.optimize import minimize
from scipy.ndimage import affine_transform
from pydicom import dcmread

#Converted to pydicom -Justin
def create_black_copy(ds_ref: pydicom.Dataset) -> pydicom.Dataset:
    # Create a deep copy of the reference DICOM dataset
    ds_new = pydicom.dcmread("black_image_template.dcm")
    
    # Copy spatial information and other relevant metadata from ds_ref to ds_new
    ds_new.PixelSpacing = ds_ref.PixelSpacing
    ds_new.ImagePositionPatient = ds_ref.ImagePositionPatient
    ds_new.ImageOrientationPatient = ds_ref.ImageOrientationPatient
    # Add other DICOM tags that you want to keep the same
    
    # Create a numpy array of zeros with the same shape as ds_ref's pixel array
    zero_pixel_array = np.zeros(ds_ref.pixel_array.shape, dtype=ds_ref.pixel_array.dtype)
    
    # Convert the numpy array to bytes and store in PixelData
    ds_new.PixelData = zero_pixel_array.tobytes()
    
    return ds_new

#converted to pydicom -Justin
def array_to_image_with_ref(data: np.ndarray, reference_image: pydicom.Dataset) -> pydicom.Dataset:
    # Convert the numpy array to a pydicom Dataset
    new_image = pydicom.dcmread("black_image_template.dcm")  # Start with a template
    
    # Update pixel data
    new_image.PixelData = data.tobytes()
    
    # Update dimensions based on the numpy array shape
    new_image.Rows, new_image.Columns = data.shape
    
    # Set the spatial information from the reference_image
    new_image.PixelSpacing = reference_image.PixelSpacing
    new_image.ImagePositionPatient = reference_image.ImagePositionPatient
    new_image.ImageOrientationPatient = reference_image.ImageOrientationPatient
    
    # Update other necessary DICOM tags
    new_image.SamplesPerPixel = 1
    new_image.PhotometricInterpretation = "MONOCHROME2"
    new_image.BitsAllocated = 16  # Assuming the numpy array is np.int16
    new_image.BitsStored = 16
    new_image.HighBit = 15
    new_image.TransferSyntaxUID = ExplicitVRLittleEndian

    return new_image

#this is the currently used
#Converted to pydicom -Justin
def scipy_register_images(target: np.ndarray, moving: np.ndarray) -> np.ndarray:
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

def test_scipy_register_images(atlas: pydicom.Dataset, image: pydicom.Dataset):
    print("testing scipy image reg")

    ds_moving_image = image  # Assuming this is already a pydicom dataset
    ds_target_image = atlas  # Assuming this is already a pydicom dataset

    moving_image = ds_moving_image.pixel_array
    target_image = ds_target_image.pixel_array

    reg_image = scipy_register_images(target_image, moving_image)

    # Create a new DICOM dataset for the registered image
    ds_reg_image = pydicom.dcmread("template.dcm")  # Assuming you have a template DICOM file to clone
    ds_reg_image.PixelData = reg_image.tobytes()
    ds_reg_image.Rows, ds_reg_image.Columns = reg_image.shape[:2]

    # Copy relevant metadata
    ds_reg_image.ImagePositionPatient = ds_moving_image.ImagePositionPatient
    ds_reg_image.ImageOrientationPatient = ds_moving_image.ImageOrientationPatient
    ds_reg_image.PixelSpacing = ds_moving_image.PixelSpacing

    data.save_pydicom_3d_img_to_dcm(ds_reg_image, "scipy_reg_image_dcm")  # Assume this function now saves a pydicom dataset
    data.save_dcm_dir_to_png_dir("scipy_reg_image_dcm", "scipy_reg_png")  

    #note: the problem may be with sitk registration where the dcm's have different values 
    # for metadata like spacing
#test_scipy_register_images("atlas", "scan2")

#expand region of interest
#this adds an extra layer of pixels to a segmented image from the original image
#converted to pydicom -Justin
def expand_roi(original_dcm: pydicom.Dataset, segment_dcm: pydicom.Dataset) -> pydicom.Dataset:
    # Convert to 3D arrays for convolution
    original_arr = original_dcm.pixel_array
    segment_arr = segment_dcm.pixel_array
    
    # Define a kernel for 3D convolution that checks for 26 neighbors in 3D
    kernel = np.ones((3, 3, 3))
    kernel[1, 1, 1] = 0
    
    # Convolve with the segment to find the boundary of ROI
    boundary = convolve(segment_arr > 0, kernel) > 0
    boundary[segment_arr > 0] = 0  # Remove areas that are already part of the segment
    
    # Create a copy of the segment
    expanded_segment_arr = segment_arr.copy()
    
    # Copy pixel values from the original image to the boundary in the expanded segment
    expanded_segment_arr[boundary] = original_arr[boundary]

    # Create a new DICOM dataset for the expanded segment
    ds_expanded_segment = pydicom.dcmread("template.dcm")  # Assuming you have a template DICOM file to clone
    ds_expanded_segment.PixelData = expanded_segment_arr.tobytes()
    ds_expanded_segment.Rows, ds_expanded_segment.Columns = expanded_segment_arr.shape[:2]
    
    # Copy relevant metadata
    ds_expanded_segment.ImagePositionPatient = segment_dcm.ImagePositionPatient
    ds_expanded_segment.ImageOrientationPatient = segment_dcm.ImageOrientationPatient
    ds_expanded_segment.PixelSpacing = segment_dcm.PixelSpacing

    #ChatGPT mentioned that these metadata fields may be necessary, but I am unconvinced so I am leaving them here, commented out, for now.
    #ds_expanded_segment.SliceThickness = segment_dcm.SliceThickness
    #ds_expanded_segment.SeriesInstanceUID = segment_dcm.SeriesInstanceUID
    #ds_expanded_segment.StudyInstanceUID = segment_dcm.StudyInstanceUID
    
    return ds_expanded_segment

# Example usage:
# original = np.random.rand(10, 10, 10)
# segment = np.zeros((10, 10, 10))
# segment[4:7, 4:7, 4:7] = 1
# result = expand_roi(original, segment)

#NOT CURRENTLY USED
# Function to calculate the Mean Squares Error between two images
# img1, img2: Two numpy arrays representing images
def mean_squares_error(img1, img2):
    # Compute the MSE and return it
    return np.sum((img1 - img2) ** 2) / img1.size

# Objective function for the optimizer to minimize
# params: List of parameters for affine transformation [scale, rotation, tx, ty]
# atlas, image: The reference and moving images as numpy arrays
def objective(params, atlas, image):
    # Decompose params into individual transformation parameters
    scale, rotation, tx, ty = params
    
    # Construct the affine transformation matrix
    matrix = np.array([[scale * np.cos(rotation), -scale * np.sin(rotation), tx],
                       [scale * np.sin(rotation), scale * np.cos(rotation), ty],
                       [0, 0, 1]])
    
    # Apply the affine transformation and create a transformed image
    transformed_image = affine_transform(image, matrix[:2, :2], (tx, ty), order=1)
    
    # Calculate the MSE between the atlas and transformed image, and return it
    return mean_squares_error(atlas, transformed_image)

# Main function for atlas-based segmentation
# atlas_file, image_file: File paths to the atlas and image DICOM files
# simMetric, optimizer, interpolator, samplerInterpolator: Parameters for registration
def atlas_segment(atlas_file, image_file, simMetric="MeanSquares", optimizer="GradientDescent", interpolator="Linear", samplerInterpolator="Linear"):
    # Read the DICOM files for the atlas and the image to be registered
    atlas_dcm = pydicom.dcmread(atlas_file)
    image_dcm = pydicom.dcmread(image_file)
    
    # Convert the DICOM pixel data to numpy arrays
    atlas = atlas_dcm.pixel_array
    image = image_dcm.pixel_array
    
    # Initial parameters for affine transformation [scale, rotation, tx, ty]
    initial_params = [1.0, 0.0, 0.0, 0.0]
    
    # Optimize the objective function to find the best transformation parameters
    result = minimize(objective, initial_params, args=(atlas, image))
    
    # Check if optimization was successful
    if result.success:
        optimized_params = result.x
    else:
        print('Optimization failed:', result.message)
        return None

    # Extract optimized parameters and build the final affine transformation matrix
    scale, rotation, tx, ty = optimized_params
    matrix = np.array([[scale * np.cos(rotation), -scale * np.sin(rotation), tx],
                       [scale * np.sin(rotation), scale * np.cos(rotation), ty],
                       [0, 0, 1]])

    # Apply the final affine transformation to the image
    registered_image = affine_transform(image, matrix[:2, :2], (tx, ty), order=1)

    # Create a new DICOM file for the registered image based on the original image's metadata
    registered_dcm = pydicom.dcmread(image_file)
    
    # Update the pixel array of the new DICOM file with the registered image
    registered_dcm.PixelData = registered_image.tobytes()
    
    return registered_dcm  # Return the registered DICOM image
# Example usage:
# registered_dcm = atlas_segment_pydicom("atlas.dcm", "image.dcm")
# if registered_dcm:
#     registered_dcm.save_as("registered.dcm")



''' def atlas_segment_pydicom(atlas_file, image_file, simMetric="MeanSquares", optimizer="GradientDescent", interpolator="Linear", samplerInterpolator="Linear"):
    
    # Read DICOM files
    atlas_dcm = pydicom.dcmread(atlas_file)
    image_dcm = pydicom.dcmread(image_file)
    
    # Get the pixel arrays as numpy arrays
    atlas = atlas_dcm.pixel_array
    image = image_dcm.pixel_array
    
    # Here, add your registration method to compute the final affine transform
    # This will replace SimpleITK's registration functionality
    # final_affine_transform = some_registration_method(atlas, image, ...)
    
    # If your final affine transform is a scipy-compatible array,
    # you can use scipy.ndimage.affine_transform to apply it:
    if interpolator == "Linear":
        mode = 'linear'
    else:
        print("Default interpolator: Linear")
        mode = 'linear'
    
    # Assuming final_affine_transform is set and is a 2D matrix
    # registered_image = affine_transform(image, final_affine_transform, mode=mode)
    
    # Create a new DICOM file for the registered image
    # You should copy most metadata from the original image
    registered_dcm = pydicom.dcmread(image_file)
    
    # Update the pixel array with the registered image
    # Assuming registered_image is a numpy array
    # registered_dcm.PixelData = registered_image.tobytes()
    
    return registered_dcm  # This would be your registered DICOM image

# You would call the function like this:
# registered_dcm = atlas_segment("atlas.dcm", "image.dcm")
# registered_dcm.save_as("registered.dcm")
'''

def test_atlas_segment_hardcoded():
    # Path to the directory that contains the DICOM files
    atlas_dir = "scan1"
    input_dir = "scan2"
    # Create 3d image with SITK
    atlas_image = data.get_3d_image(atlas_dir)
    input_image = data.get_3d_image(input_dir)
    #does it need to by cast to float32?

    study_id = input_image.GetMetaData('0020|000D') if input_image.HasMetaDataKey('0020|000D') else ""
    series_id = input_image.GetMetaData('0020|000E') if input_image.HasMetaDataKey('0020|000E') else ""
    print("study id: ", study_id)
    print("series id: ", series_id)

    registered_image = atlas_segment(atlas_image, input_image)

    #data.view_sitk_3d_image(map_image, 5, "map image")
    #data.view_sitk_3d_image(input_image, 5, "input image")
    #data.view_sitk_3d_image(registered_image, 5, "registered image")

    data.save_sitk_3d_img_to_dcm(registered_image, "registered")
    data.save_dcm_dir_to_png_dir("registered", "reg pngs")
#test_atlas_segment_hardcoded()


# Converted to pydicom. I am less certain about this pydicom change than the others. -Justin
def initial_segment_test():
    # Load DICOM filepaths
    dicom_path1 = "scan2/ADNI_003_S_1059_PT_adni2__br_raw_20071211125948781_11_S43552_I84553.dcm"
    dicom_path2 = "scan2/ADNI_003_S_1059_PT_adni2__br_raw_20071211125949312_18_S43552_I84553.dcm"

    # Read DICOM files using pydicom
    dicom1 = pydicom.dcmread(dicom_path1)
    dicom2 = pydicom.dcmread(dicom_path2)

    # Convert DICOM pixel data to numpy arrays
    pixel_array1 = dicom1.pixel_array
    pixel_array2 = dicom2.pixel_array

    # Perform image registration using our custom atlas_segment function
    registered_dcm = atlas_segment(dicom_path1, dicom_path2)

    # Exit if optimization failed
    if registered_dcm is None:
        print("Registration failed.")
        return

    # Convert the registered DICOM pixel data to a numpy array
    registered_array = registered_dcm.pixel_array

    # Plot the original and registered images
    plt.figure(figsize=(15, 5))

    # Plot Image 1 - Before Registration
    plt.subplot(131)
    plt.imshow(pixel_array1.squeeze(), cmap='gray')
    plt.title("Image 1 - Before Registration")
    plt.axis('off')

    # Plot Image 2 - Before Registration
    plt.subplot(132)
    plt.imshow(pixel_array2.squeeze(), cmap='gray')
    plt.title("Image 2 - Before Registration")
    plt.axis('off')

    # Plot Registered Image
    plt.subplot(133)
    plt.imshow(registered_array.squeeze(), cmap='gray')
    plt.title("Registered Image")
    plt.axis('off')

    plt.show()

# Run the function
#initial_segment_test()

#data.save_dcm_dir_to_png_dir("atlas", "atlas pngs")
#data.save_dcm_dir_to_png_dir("registered", "reg pngs")

#THIS FUNCTION WILL BE DEPRECATED SOON, AS THERE IS A FUNCTION IN THE DATA MODULE THAT DOES IT MORE SIMPLY
#given a dictionary with region names as keys and sitk images as values, this funciton displays them
#I have not any changes to this function because the above comments suggest that this function should be deleted anyway. -Justin
def display_regions_from_dict(region_images):
    for region_name, region_image in region_images.items():
        print(region_name)
        print(region_image.GetSize())

        plt.figure(figsize=(6, 6))
        array_from_image = sitk.GetArrayFromImage(region_image)
            # Displaying the first slice of the 3D image
        plt.imshow(array_from_image[0, :, :], cmap='gray')
        plt.axis('off')
        plt.title(f"Region: {region_name}")
        plt.show()

#converted to pydicom -Justin
def create_seg_images_from_image(images_dict, coords_dict):
    output_images = {}
    
    for region_name, coordinates_list in coords_dict.items():
        if region_name not in images_dict:
            print(f"Warning: No image found for region {region_name}")
            continue

        current_image = images_dict[region_name]
        current_pixel_array = current_image.pixel_array
        
        # Create a black copy of the current image
        blank_image = create_black_copy(current_image)
        blank_pixel_array = blank_image.pixel_array
        
        for coordinates in coordinates_list:
            x, y, z = coordinates
            if (0 <= x < current_pixel_array.shape[0]) and \
               (0 <= y < current_pixel_array.shape[1]) and \
               (0 <= z < current_pixel_array.shape[2]):
                pixel_value = current_pixel_array[x, y, z]
                blank_pixel_array[x, y, z] = pixel_value

        # Update pixel data
        blank_image.PixelData = blank_pixel_array.tobytes()
        output_images[region_name] = blank_image
        
    print(f"Size of output images:  {len(output_images)}")
    return output_images


#this would take a dict of atlas segmented images, and then further refine them with coordinates output by an 
# Advanced Segmentation algo, with corresponding region names

#converted to pydicom -Justin
def create_seg_images_from_dict(images_dict, coords_dict):
    output_images = {}

    for region_name, coordinates_list in coords_dict.items():
        # Ensure that the region_name exists in the images_dict
        if region_name not in images_dict:
            print(f"Warning: No image found for region {region_name}")
            continue

        current_image = images_dict[region_name]
        
        # Extract the numpy array from the current DICOM dataset
        current_pixel_array = current_image.pixel_array

        # Create a black copy of the current image
        blank_image = create_black_copy(current_image)

        # Extract the numpy array from the blank DICOM dataset
        blank_pixel_array = blank_image.pixel_array

        for coordinates in coordinates_list:
            x, y, z = coordinates
            if (0 <= x < current_pixel_array.shape[0]) and \
               (0 <= y < current_pixel_array.shape[1]) and \
               (0 <= z < current_pixel_array.shape[2]):
                pixel_value = current_pixel_array[x, y, z]
                blank_pixel_array[x, y, z] = pixel_value

        # Update the pixel data in the blank_image DICOM dataset
        blank_image.PixelData = blank_pixel_array.tobytes()

        # Append the finished blank_image to the output_images dictionary
        output_images[region_name] = blank_image

    print(f"Size of output images:  {len(output_images)}")

    return output_images


def DCMs_to_pydicom_img_dict(directory):
    image = data.get_3d_image(directory)  
    
    def generate_regions():
        region1 = [[x, y, z] for x in range(0, 51) for y in range(0, 51) for z in range(0, 51)]
        region2 = [[x, y, z] for x in range(50, 101) for y in range(50, 101) for z in range(0, 50)]
        
        region_dict = {
            "Region1": region1,
            "Region2": region2
        }
        
        return region_dict
    
    region_dict = generate_regions()
    region_images = create_seg_images_from_image(image, region_dict)  
    data.display_seg_images(region_images)  

    

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

def encode_atlas_colors(image_list: list) -> dict:
    color_to_region_dict = {
        (237, 28, 36): 'Skull',
        (0, 162, 232): 'Brain',
    }

    region_coords_dict = {region: [] for region in color_to_region_dict.values()}

    for z, image_2d in enumerate(image_list):
        for y in range(image_2d.shape[0]):
            for x in range(image_2d.shape[1]):
                pixel_color = tuple(image_2d[y, x])
                max_channel = max(pixel_color)
                
                if max_channel > 100:
                    if pixel_color.index(max_channel) == 0:
                        pixel_color = (237, 28, 36)
                    elif pixel_color.index(max_channel) == 2:
                        pixel_color = (0, 162, 232)
                        
                if pixel_color in color_to_region_dict:
                    region = color_to_region_dict[pixel_color]
                    region_coords_dict[region].append([x, y, z])
    
    return region_coords_dict


#converted to pydicom -Justin
#atlas and image are both pydicom Datasets, atlas_colors is a list of 2d np arrays
def execute_atlas_seg(atlas, atlas_colors, image):
    print("executing atlas seg")
    # Convert the pydicom Datasets to NumPy arrays
    moving_image = image.pixel_array
    target_image = atlas.pixel_array

    #register the 3d array to atlas 3d array
    reg_image_array = scipy_register_images(target_image, moving_image)

    #convert registered array to pydicom Dataset
    reg_image = array_to_pydicom_with_ref(reg_image_array, image)

    #coordinates of region based on atlas
    region_to_coord_dict = encode_atlas_colors(atlas_colors)

    #create image dict from coords dict and pydicom Dataset
    final_dict = create_seg_images_from_image(reg_image, region_to_coord_dict)

    #expand roi
    for region, segment in final_dict.items():
        final_dict[region] = expand_roi(reg_image, segment)

    return final_dict
#RuntimeError: filter weights array has incorrect shape.

def array_to_pydicom_with_ref(np_array, ref_dicom):
    # Create a new DICOM object based on the reference DICOM
    new_dicom = pydicom.dcmread(ref_dicom.filename)
    
    # Update the pixel array
    new_dicom.PixelData = np_array.tobytes()
    
    # Update the relevant metadata (adjust as needed)
    new_dicom.Rows, new_dicom.Columns = np_array.shape[:2]
    
    if len(np_array.shape) == 3:
        new_dicom.NumberOfFrames = np_array.shape[0]
    
    return new_dicom

# atlas and image are both pydicom Datasets, atlas_colors is a list of 2d np arrays
def execute_atlas_seg(atlas, atlas_colors, image):
    print("executing atlas seg")

    # CHANGED: Converted from SimpleITK Image to pydicom Dataset
    moving_image = image.pixel_array
    target_image = atlas.pixel_array

    # Register the 3D array to atlas 3D array
    reg_image_array = scipy_register_images(target_image, moving_image)

    # CHANGED: Converted from SimpleITK Image to pydicom Dataset
    reg_image = array_to_pydicom_with_ref(reg_image_array, image)

    # Coordinates of region based on atlas
    region_to_coord_dict = encode_atlas_colors(atlas_colors)

    # CHANGED: Converted from SimpleITK Image to pydicom Dataset
    final_dict = create_seg_images_from_image(reg_image, region_to_coord_dict)

    # Expand ROI
    for region, segment in final_dict.items():
        final_dict[region] = expand_roi(reg_image, segment)

    return final_dict


if __name__ == "__main__":
    atlas_path = data.get_atlas_path()  # Should return a path to a DICOM file for the atlas
    atlas = pydicom.dcmread(atlas_path)  # Read the DICOM file into a pydicom Dataset object

    image_path = "scan1"  # Assuming this is a DICOM file path
    image = pydicom.dcmread(image_path)  # Read the DICOM file into a pydicom Dataset object

    color_atlas = data.get_2d_png_array_list("color atlas")  # Assuming this still returns a list of 2D numpy arrays

    # Execute the modified atlas segmentation function that should now work with pydicom Datasets
    seg_results = execute_atlas_seg(atlas, color_atlas, image)
    
    # If `data.store_seg_img_on_file` expects SimpleITK images, it should be modified to handle pydicom Datasets
    data.store_seg_img_on_file(seg_results, "seg results test")

    coords_dict = {
        "Brain": [(x, y, z) for x in range(64) for y in range(128) for z in range(46)]
    }

    # Create segmented images from the coordinates dictionary
    seg_dict_from_seg_dict = create_seg_images_from_dict(seg_results, coords_dict)

    # Store the segmented images
    data.store_seg_img_on_file(seg_dict_from_seg_dict, "seg from seg test")


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