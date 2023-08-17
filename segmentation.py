#dicom_path2 = "ADNI_007_S_1339_PT_TRANSAXIAL_BRAIN_3D_FDG_ADNI_CTAC__br_raw_20070402142342176_8_S29177_I47688.dcm"
#dicom_path1 = "ADNI_007_S_1339_PT_TRANSAXIAL_BRAIN_3D_FDG_ADNI_CTAC__br_raw_20070402142342697_9_S29177_I47688.dcm"

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import data


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
    initial_transform = sitk.TranslationTransform(atlas.GetDimension())
    registration_method.SetInitialTransform(initial_transform)

    #set interpolator
    if interpolator == "Linear":
        registration_method.SetInterpolator(sitk.sitkLinear)
    else:
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

data.save_dcm_dir_to_png_dir("scan1", "scan 1")
data.save_dcm_dir_to_png_dir("scan2", "scan 2")


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