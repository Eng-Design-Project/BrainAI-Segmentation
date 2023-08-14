import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import data

# Define the patch size and stride (how much patches overlap)
patch_size = (64, 64, 64)  # Adjust the size according to your model's input requirements
stride = (32, 32, 32)  # Adjust the stride for overlap

# Function to copy pixel data from input image to a new image based on region coordinates
def get_region(image, region_coordinates):
    # Create a new image with the same properties as the input image
    new_image = sitk.Image(image.GetSize(), image.GetPixelIDValue())
    new_image.SetSpacing(image.GetSpacing())
    new_image.CopyInformation(image)
    
    # Set all pixels in the new image to black (zero)
    new_image_array = sitk.GetArrayFromImage(new_image)
    new_image_array.fill(0)
    
    # Copy pixel data from input image to new image based on coordinates
    for coord in region_coordinates:
        x, y, z = coord
        pixel_value = image.GetPixel((x, y, z))
        new_image.SetPixel((x, y, z), pixel_value)
        
    return new_image

# Load DICOM filepaths
dicom_path1 = "scan1"
#dicom_path2 = "scan2/ADNI_003_S_1059_PT_adni2__br_raw_20071211125949312_18_S43552_I84553.dcm"

image = data.get_3d_image(dicom_path1)

# Define region_coordinates based on your requirements
region_coordinates = [[0,0,0]]  # Replace with actual coordinates

width, height, depth = image.GetSize()

for x in range(width):
    for y in range(height):
        for z in range(depth):
            region_coordinates.append((x, y, z))

# Call the get_region function to create a new image with the specified region
new_image = get_region(image, region_coordinates)

# Display the region image
plt.imshow(sitk.GetArrayFromImage(new_image)[0, :, :], cmap='gray')
plt.axis('off')
plt.title("Segmented Brain Region")
plt.show()





