import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

# Define the patch size
patch_size = (64, 64, 64)

def create_image_from_regions(image, region_dict, patch_size):
    half_patch_size = [size // 2 for size in patch_size]
    output_images = {}
    
    for region_name, coordinates_list in region_dict.items():
        blank_image = sitk.Image(image.GetSize(), image.GetPixelID())
        blank_image.SetSpacing(image.GetSpacing())
        blank_image.SetOrigin(image.GetOrigin())
        
        for coordinates in coordinates_list:
            x, y, z = coordinates
            if (0 <= x < image.GetSize()[0]) and \
               (0 <= y < image.GetSize()[1]) and \
               (0 <= z < image.GetSize()[2]):
                region = image[x - half_patch_size[0]:x + half_patch_size[0],
                               y - half_patch_size[1]:y + half_patch_size[1],
                               z - half_patch_size[2]:z + half_patch_size[2]]
                
                # Debug: print the unique pixel values in the region
                print(f"Unique pixel values in the region at coordinates {coordinates}: {np.unique(sitk.GetArrayFromImage(region))}")

                blank_image = sitk.Paste(blank_image, region, region.GetSize(), [0, 0, 0], coordinates)
        
        output_images[region_name] = blank_image

    return output_images

# Load DICOM image series
dicom_series_path = "scan1"
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(dicom_series_path)
reader.SetFileNames(dicom_names)
image = reader.Execute()

# Define your regions and their coordinates here
region_dict = {
    "Region1": [[40, 40, 40], [80, 80, 80]],
    "Region2": [[120, 120, 120], [160, 160, 160]]
}

# Create images from the regions
region_images = create_image_from_regions(image, region_dict, patch_size)

# Display each valid region image in a separate window
for region_name, region_image in region_images.items():
    if region_image.GetNumberOfPixels() > 0:
        plt.figure(figsize=(6, 6))
        array_from_image = sitk.GetArrayFromImage(region_image)
        if array_from_image.any():
            plt.imshow(array_from_image[0, :, :], cmap='gray')
            plt.axis('off')
            plt.title(f"Region: {region_name}")
            plt.show()
        else:
            print(f"Region: {region_name} is all black.")




