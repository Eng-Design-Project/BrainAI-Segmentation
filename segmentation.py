#dicom_path2 = "ADNI_007_S_1339_PT_TRANSAXIAL_BRAIN_3D_FDG_ADNI_CTAC__br_raw_20070402142342176_8_S29177_I47688.dcm"
#dicom_path1 = "ADNI_007_S_1339_PT_TRANSAXIAL_BRAIN_3D_FDG_ADNI_CTAC__br_raw_20070402142342697_9_S29177_I47688.dcm"

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import data

import numpy as np
from scipy.ndimage import affine_transform
from scipy.signal import fftconvolve
import SimpleITK as sitk

def create_black_copy(image: sitk.Image) -> sitk.Image:
    # Create a copy of the input image
    black_image = sitk.Image(image.GetSize(), image.GetPixelID())
    black_image.SetOrigin(image.GetOrigin())
    black_image.SetSpacing(image.GetSpacing())
    black_image.SetDirection(image.GetDirection())

    # All pixel values are already set to 0 (black) upon initialization
    return black_image

#for expand region of interest
from scipy.ndimage import convolve

def array_to_image_with_ref(data: np.ndarray, reference_image: sitk.Image) -> sitk.Image:
    # Convert the numpy array to a SimpleITK image
    new_image = sitk.GetImageFromArray(data)
    
    # Set the spatial information from the reference_image
    new_image.SetSpacing(reference_image.GetSpacing())
    new_image.SetOrigin(reference_image.GetOrigin())
    new_image.SetDirection(reference_image.GetDirection())

    return new_image

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
    #get sitk images
    sitk_moving_image = data.get_3d_image(image)
    sitk_target_image = data.get_3d_image(atlas)
    # Convert the SimpleITK Images to NumPy arrays
    moving_image = sitk.GetArrayFromImage(sitk_moving_image)
    target_image = sitk.GetArrayFromImage(sitk_target_image)
    #register the 3d array to atlas 3d array
    reg_image = scipy_register_images(target_image, moving_image)
    #convert registered array to sitk image
    reg_image = array_to_image_with_ref(reg_image, sitk_moving_image)
    #save registered image as dcm first, then as png
    data.save_sitk_3d_img_to_dcm(reg_image, "scipy_reg_image_dcm")
    data.save_dcm_dir_to_png_dir("scipy_reg_image_dcm", "scipy_reg_png")

    #note: the problem may be with sitk registration where the dcm's have different values 
    # for metadata like spacing
#test_scipy_register_images("atlas", "scan2")

# Example usage
#target_image = np.random.rand(100, 100, 100)
#moving_image = np.roll(target_image, shift=5, axis=0)  # Let's shift target image by 5 pixels in x-direction
#registered_image = scipy_register_images(target_image, moving_image)
# Now, the 'registered_image' should be closely aligned with 'target_image'


#expand region of interest
#this adds an extra layer of pixels to a segmented image from the original image
#currently takes 2 3d arrays. Unsure if this will take simpleITK images instead
def expand_roi(original, segment):
    # Define a kernel for 3D convolution that checks for 26 neighbors in 3D
    kernel = np.ones((3, 3, 3))
    kernel[1, 1, 1] = 0
    
    # Convolve with the segment to find the boundary of ROI
    boundary = convolve(segment > 0, kernel) > 0
    boundary[segment > 0] = 0  # Remove areas that are already part of the segment
    
    # Create a copy of the segment
    expanded_segment = segment.copy()
    
    # Copy pixel values from the original image to the boundary in the expanded segment
    expanded_segment[boundary] = original[boundary]
    
    return expanded_segment

# Example usage:
# original = np.random.rand(10, 10, 10)
# segment = np.zeros((10, 10, 10))
# segment[4:7, 4:7, 4:7] = 1
# result = expand_roi(original, segment)


def atlas_segment(atlas, image, 
                  simMetric="MeanSquares", optimizer="GradientDescent", 
                  interpolator="Linear", samplerInterpolator="Linear"):

    #set up the registration framework
    registration_method = sitk.ImageRegistrationMethod()

    #set similarity metric
    if simMetric == "MeanSquares":
        registration_method.SetMetricAsMeanSquares()
    else:
        print("default sim metric: MeanSquares")
        registration_method.SetMetricAsMeanSquares()

    #set optimizer
    if optimizer == "GradientDescent":
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        registration_method.SetOptimizerScalesFromPhysicalShift()
    else:
        print("default optimizer: GradientDescent")
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        registration_method.SetOptimizerScalesFromPhysicalShift()

    #initial transform
    #initial_transform = sitk.TranslationTransform(atlas.GetDimension())
        #transforms only translation? affline instead?
    #Rigid Transform (Rotation + Translation):
    #initial_transform = sitk.Euler3DTransform()
    #Similarity Transform (Rigid + isotropic scaling):
    #initial_transform = sitk.Similarity3DTransform()
    #Affine Transform (includes rotations, translations, scaling, and shearing):
    initial_transform = sitk.AffineTransform(atlas.GetDimension())
    #BSpline Transform (a non-rigid, deformable transform): *DOES NOT CURRENTLY WORK*
    #order_x, order_y, order_z = 5, 5, 5
    #initial_transform = sitk.BSplineTransformInitializer(atlas, [order_x, order_y, order_z])


    registration_method.SetInitialTransform(initial_transform)

    #set interpolator
    if interpolator == "Linear":
        registration_method.SetInterpolator(sitk.sitkLinear)
    else:
        print("default interpolator: Linear")
        registration_method.SetInterpolator(sitk.sitkLinear)

    #execute registration
    final_transform = registration_method.Execute(sitk.Cast(atlas, sitk.sitkFloat32), sitk.Cast(image, sitk.sitkFloat32))

    #apply transformation
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(atlas)
    if samplerInterpolator == "Linear":
        resampler.SetInterpolator(sitk.sitkLinear)
    elif samplerInterpolator == "HigherOrder":
        resampler.SetInterpolator(sitk.sitkBSpline)
    elif samplerInterpolator == "NearestNeighbor":
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        print("default samplerInterpolator Linear")
        resampler.SetInterpolator(sitk.sitkLinear)
    
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(final_transform)

    registered_image = resampler.Execute(image)
    return registered_image

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

def initial_segment_test():
    # Load DICOM filepaths
    dicom_path1 = "scan2/ADNI_003_S_1059_PT_adni2__br_raw_20071211125948781_11_S43552_I84553.dcm"
    dicom_path2 = "scan2/ADNI_003_S_1059_PT_adni2__br_raw_20071211125949312_18_S43552_I84553.dcm"

    # get sitk images
    image1 = sitk.ReadImage(dicom_path1, sitk.sitkFloat32)
    image2 = sitk.ReadImage(dicom_path2, sitk.sitkFloat32)

    # Convert SimpleITK images to numpy arrays for displaying
    pixel_array1 = sitk.GetArrayFromImage(image1)
    pixel_array2 = sitk.GetArrayFromImage(image2)

    # Print the dimensions of image1 and image2 before registration
    print("Image 1 - Before Registration:")
    print("Size:", image1.GetSize())
    print("Spacing:", image1.GetSpacing())
    print("Direction:", image1.GetDirection())
    print()

    print("Image 2 - Before Registration:")
    print("Size:", image2.GetSize())
    print("Spacing:", image2.GetSpacing())
    print("Direction:", image2.GetDirection())
    print()

    # Perform image registration using Demons registration filter
    demons_registration_filter = sitk.DemonsRegistrationFilter()
    final_transform = demons_registration_filter.Execute(image1, image2)

    # Explicitly set the transformation type to DisplacementFieldTransform
    final_transform = sitk.DisplacementFieldTransform(final_transform)

    # Apply the final transformation to image2 (moving image)
    registered_image = sitk.Resample(image2, image1, final_transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)

    # Convert SimpleITK images to numpy arrays for displaying
    pixel_array3 = sitk.GetArrayFromImage(image1)
    pixel_array4 = sitk.GetArrayFromImage(image2)
    registered_array = sitk.GetArrayFromImage(registered_image)

    # Plot the original and registered images
    plt.figure(figsize=(20, 5))

    # Plot Image 1 - Before Registration
    plt.subplot(151)
    plt.imshow(pixel_array1.squeeze(), cmap='gray')
    plt.title("Image 1 - Before Registration")
    plt.axis('off')

    # Plot Image 2 - Before Registration
    plt.subplot(152)
    plt.imshow(pixel_array2.squeeze(), cmap='gray')
    plt.title("Image 2 - Before Registration")
    plt.axis('off')

    # Plot Image 1 - After Registration
    plt.subplot(153)
    plt.imshow(pixel_array3.squeeze(), cmap='gray')
    plt.title("Image 1 - After Registration")
    plt.axis('off')

    # Plot Image 2 - After Registration
    plt.subplot(154)
    plt.imshow(pixel_array4.squeeze(), cmap='gray')
    plt.title("Image 2 - After Registration")
    plt.axis('off')

    # Plot Registered Image
    plt.subplot(155)
    plt.imshow(registered_array.squeeze(), cmap='gray')
    plt.title("Registered Image")
    plt.axis('off')

    plt.show()

#data.save_dcm_dir_to_png_dir("atlas", "atlas pngs")
#data.save_dcm_dir_to_png_dir("registered", "reg pngs")

#THIS FUNCTION WILL BE DEPRECATED SOON, AS THERE IS A FUNCTION IN THE DATA MODULE THAT DOES IT MORE SIMPLY
#given a dictionary with region names as keys and sitk images as values, this funciton displays them
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

def create_seg_images(image, region_dict):
    output_images = {}
    for region_name, coordinates_list in region_dict.items():
        blank_image = create_black_copy(image)
        
        for coordinates in coordinates_list:
            x, y, z = coordinates
            if (0 <= x < image.GetSize()[0]) and \
               (0 <= y < image.GetSize()[1]) and \
               (0 <= z < image.GetSize()[2]):
                pixel_value = image[x, y, z]
                blank_image[x, y, z] = pixel_value
                
        # Append the finished blank_image to the output_images dictionary
        output_images[region_name] = blank_image

    print(f"Size of output images:  {len(output_images)}")

    return output_images

# takes a directory of DCMs, outputs a dictionary with region names as keys and sitk images as the values
def DCMs_to_sitk_img_dict(directory):
    image = data.get_3d_image(directory)

    #this part of the function could be expanded to have more regions
    def generate_regions(): 
        region1 = [[x, y, z] for x in range(0, 51) for y in range(0, 51) for z in range(0, 51)]
        region2 = [[x, y, z] for x in range(50, 101) for y in range(50, 101) for z in range(0, 50)]

        region_dict = {
            "Region1": region1,
            "Region2": region2
        }
        return region_dict
    
    # Define your regions and their coordinates here
    region_dict = generate_regions()
    region_images = create_seg_images(image, region_dict)
    #display_regions_from_dict(region_images)
    data.display_seg_images(region_images)
    
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

def encode_atlas_colors(image_3d: np.ndarray) -> dict:
    #hard coded colors to region
    color_to_region_dict = {
        (255, 0, 0): 'Region1',   # Red
        (0, 0, 255): 'Region2',   # Blue
        (0, 255, 0): 'Region3',   # Green
        #... add other colors and regions as required
    }
    # Initialize the output dictionary with region names as keys and empty lists as values
    region_coords_dict = {region: [] for region in color_to_region_dict.values()}

    # Iterate through the 3D array to get the coordinates and pixel values
    for x in range(image_3d.shape[0]):
        for y in range(image_3d.shape[1]):
            for z in range(image_3d.shape[2]):
                pixel_color = tuple(image_3d[x, y, z])#note, xyz from 3d arrays
                # If the pixel color exists in the dictionary, add its coordinate to the respective list
                if pixel_color in color_to_region_dict:
                    region = color_to_region_dict[pixel_color]
                    region_coords_dict[region].append([z, y, x])#note zyx for comparison to sitk images

    return region_coords_dict

def test_encode_atlas_colors():
    # Define image dimensions
    width = 40
    height = 40
    depth = 40
    # Create an empty RGB 3D image
    image_3d = np.zeros((depth, height, width, 3), dtype=np.uint8)
    # Fill the left with red
    image_3d[:, :, :width//3] = [255, 0, 0]
    # Fill the middle with blue
    image_3d[:, :, width//3:2*width//3] = [0, 0, 255]
    # Fill the right with green
    image_3d[:, :, 2*width//3:] = [0, 255, 0]
    #get a dict of regions and coords
    region_to_coord_dict = encode_atlas_colors(image_3d)
    #print(region_to_coord_dict)
    #convert 3d array image to sitk image
    sitk_image = sitk.GetImageFromArray(image_3d)
    data.view_sitk_3d_image(sitk_image, 5, "redbluegreen")

    #create image dict from coords dict and sitk_image
    final_dict = create_seg_images(sitk_image, region_to_coord_dict)

    #test expand_roi()
    np_original = sitk.GetArrayFromImage(sitk_image)
    #following is code to test expand ROI, but it doesn't work on RGB images
    # this is fine, we only need it to work on grayscale
    # for region, image in final_dict.items():
    #     np_image = sitk.GetArrayFromImage(image)
    #     for x in range(10):
    #         #RuntimeError: filter weights array has incorrect shape.
    #         np_image = expand_roi(np_original, np_image)
    #     final_dict[region] = sitk.GetImageFromArray(np_image)
    #data.display_seg_images(final_dict)
    print(image_3d.shape)
    print(sitk_image.GetSize())
    print(np_original.shape)
    np_image = sitk.GetArrayFromImage(final_dict["Region1"])
    print(np_image.shape)
#RuntimeError: filter weights array has incorrect shape.
#test_encode_atlas_colors()

#atlas and image are both sitk images, atlas colors is a 3d np array
def execute_atlas_seg(atlas, atlas_colors, image):
    print("executing atlas seg")
    # Convert the SimpleITK Images to NumPy arrays
    moving_image = sitk.GetArrayFromImage(image)
    target_image = sitk.GetArrayFromImage(atlas)
    #register the 3d array to atlas 3d array
    reg_image_array = scipy_register_images(target_image, moving_image)
    #convert registered array to sitk image
    reg_image = array_to_image_with_ref(reg_image_array, image)
    
    #coordinates of region based on atlas
    region_to_coord_dict = encode_atlas_colors(atlas_colors)

    #create image dict from coords dict and sitk_image
    final_dict = create_seg_images(reg_image, region_to_coord_dict)

    #expand roi
    for region, segment in final_dict.items():
        final_dict[region] = expand_roi(reg_image, segment)

    return final_dict


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