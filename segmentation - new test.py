import SimpleITK as sitk
import pydicom
import os

# Path to the directory that contains the DICOM files
directory1 = "scan2"
#directory2 = "scan2"

reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(directory1)
reader.SetFileNames(dicom_names)
image = reader.Execute()

size = image.GetSize()
print("Image size:", size)

import matplotlib.pyplot as plt

# Let's say 'image' is your 3D SimpleITK image
array = sitk.GetArrayFromImage(image)

# Decide on the number of slices you want to visualize
num_slices = 100

# Calculate the step size
step = array.shape[0] // num_slices

# Generate the slices
slices = [array[i*step, :, :] for i in range(num_slices)]

# Display the slices
fig, axes = plt.subplots(1, num_slices, figsize=(18, 18))
for i, slice in enumerate(slices):
    axes[i].imshow(slice, cmap='gray')
    axes[i].axis('off')
plt.show()


# Loop over slices
'''
for z in range(size[2]):
    slice = image[:,:,z]
    position = slice.GetMetaData('0020|0032')  # Image Position (Patient) tag
    print("Slice ", z, " position: ", position)
'''

# Get a list of all DICOM files in the directory
scan1_files = [os.path.join(directory1, f) for f in os.listdir(directory1) if f.endswith(".dcm")]
#scan2_files = [os.path.join(directory2, f) for f in os.listdir(directory2) if f.endswith(".dcm")]

'''
for filename in scan1_files:
    image = sitk.ReadImage(filename)
    print(image.GetMetaData("0020|0032"))
'''

# Read in the image series
image1 = sitk.ReadImage(scan1_files)
#image2 = sitk.ReadImage(scan2_files)

# Now 'image' is a 3D SimpleITK.Image object that represents the full volume


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


