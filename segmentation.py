#dicom_path2 = "ADNI_007_S_1339_PT_TRANSAXIAL_BRAIN_3D_FDG_ADNI_CTAC__br_raw_20070402142342176_8_S29177_I47688.dcm"
#dicom_path1 = "ADNI_007_S_1339_PT_TRANSAXIAL_BRAIN_3D_FDG_ADNI_CTAC__br_raw_20070402142342697_9_S29177_I47688.dcm"

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

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