import pydicom
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

# Load DICOM files
dicom_path1 = "ADNI_007_S_1339_PT_TRANSAXIAL_BRAIN_3D_FDG_ADNI_CTAC__br_raw_20070402142342697_9_S29177_I47688.dcm"
dicom_path2 = "ADNI_007_S_1339_PT_TRANSAXIAL_BRAIN_3D_FDG_ADNI_CTAC__br_raw_20070402142344709_12_S29177_I47688.dcm"

dicom1 = pydicom.dcmread(dicom_path1)
dicom2 = pydicom.dcmread(dicom_path2)

# Extract pixel arrays from DICOM files
pixel_array1 = dicom1.pixel_array
pixel_array2 = dicom2.pixel_array

# Convert pixel arrays to SimpleITK images
image1 = sitk.GetImageFromArray(pixel_array1.astype(np.float32))
image2 = sitk.GetImageFromArray(pixel_array2.astype(np.float32))

# Print the dimensions of image1 and image2 before resampling
print("Image 1 - Before Resampling:")
print("Size:", image1.GetSize())
print("Spacing:", image1.GetSpacing())
print("Direction:", image1.GetDirection())
print()

print("Image 2 - Before Resampling:")
print("Size:", image2.GetSize())
print("Spacing:", image2.GetSpacing())
print("Direction:", image2.GetDirection())
print()

# Get the size and spacing of image1
size_image1 = image1.GetSize()
spacing_image1 = image1.GetSpacing()

# Resample image2 to match the size and spacing of image1
resampler = sitk.ResampleImageFilter()
resampler.SetSize(size_image1)
resampler.SetOutputSpacing(spacing_image1)
resampled_image2 = resampler.Execute(image2)
image2 = resampled_image2

# Print the dimensions of image1 and image2 after resampling
print("Image 1 - After Resampling:")
print("Size:", image1.GetSize())
print("Spacing:", image1.GetSpacing())
print("Direction:", image1.GetDirection())
print()

print("Image 2 - After Resampling:")
print("Size:", image2.GetSize())
print("Spacing:", image2.GetSpacing())
print("Direction:", image2.GetDirection())
print()

# Initialize the registration with an identity transformation
initial_transform = sitk.CenteredTransformInitializer(image1, image2, sitk.CenteredTransformInitializerFilterType.PLAIN)

# Perform image registration
registration_method = sitk.ImageRegistrationMethod()

# Set the Demons similarity metric
registration_method.SetMetricAsDemons(10)  # The smoothing parameter can be adjusted, higher values are more smooth

# Set the optimizer
registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)

# Set the interpolator
registration_method.SetInterpolator(sitk.sitkLinear)

# Execute the registration with the initial transform
final_transform = registration_method.Execute(image1, image2, initial_transform)

# Apply the final transformation to image2
registered_image = sitk.Resample(image2, image1, final_transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)

# Convert SimpleITK image back to numpy array
registered_array = sitk.GetArrayFromImage(registered_image)

# Plot the original and registered images
plt.figure(figsize=(10, 5))
plt.subplot(131)
plt.imshow(pixel_array1, cmap='gray')
plt.title("Image 1")
plt.axis('off')

plt.subplot(132)
plt.imshow(pixel_array2, cmap='gray')
plt.title("Image 2")
plt.axis('off')

plt.subplot(133)
plt.imshow(registered_array, cmap='gray')
plt.title("Registered Image")
plt.axis('off')

plt.show()















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
    
    sitk image copied into numpy array
    npy_array = sitk.GetArrayFromImage(image)

    np array to sitk image
    new_image = sitk.GetImageFromArray(npy_array)


'''
  