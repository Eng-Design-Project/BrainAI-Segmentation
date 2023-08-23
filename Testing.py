import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

# Define the patch size and stride (how much patches overlap)
patch_size = (64, 64, 64)  # Adjust the size according to your model's input requirements
stride = (32, 32, 32)  # Adjust the stride for overlap

# Function to split an image into regions based on coordinates
def split_image_into_regions(image, region_coordinates_list, patch_size):
    region_images = []
    half_patch_size = [size // 2 for size in patch_size]
    
    for coordinates in region_coordinates_list:
        x, y, z = coordinates
        if (0 <= x < image.GetSize()[0]) and \
           (0 <= y < image.GetSize()[1]) and \
           (0 <= z < image.GetSize()[2]):
            region = image[x - half_patch_size[0]:x + half_patch_size[0],
                           y - half_patch_size[1]:y + half_patch_size[1],
                           z - half_patch_size[2]:z + half_patch_size[2]]
            region_images.append(region)
    
    return region_images

# Load DICOM image series
dicom_series_path = "scan1"
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(dicom_series_path)
reader.SetFileNames(dicom_names)
image = reader.Execute()

# Define region_coordinates based on your requirements
region_coordinates_list = [[x, y, z] for x in range(0, image.GetSize()[0], stride[0])
                                       for y in range(0, image.GetSize()[1], stride[1])
                                       for z in range(0, image.GetSize()[2], stride[2])]

# Split the image into regions
region_images = split_image_into_regions(image, region_coordinates_list, patch_size)

# Display each valid region image in a separate window
for i, region in enumerate(region_images):
    if region.GetNumberOfPixels() > 0:  # Display only if the region has valid pixels
        plt.figure(figsize=(6, 6))
        plt.imshow(sitk.GetArrayFromImage(region)[0, :, :], cmap='gray')
        plt.axis('off')
        plt.title(f"Segmented Brain Region {i+1}")
        plt.show()





