import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import data

# Define the patch size
def create_black_copy(image: sitk.Image) -> sitk.Image:
    # Create a copy of the input image
    black_image = sitk.Image(image.GetSize(), image.GetPixelID())
    black_image.SetOrigin(image.GetOrigin())
    black_image.SetSpacing(image.GetSpacing())
    black_image.SetDirection(image.GetDirection())

    # All pixel values are already set to 0 (black) upon initialization
    return black_image

def create_image_from_regions(image, region_dict):
    output_images = {}
    #my_dict['key3'] = 'value3'
    for region_name, coordinates_list in region_dict.items():
        blank_image = image
        blank_image = create_black_copy(image)
        blank_image.SetOrigin(image.GetOrigin())
        
        
        for coordinates in coordinates_list:
            x, y, z = coordinates
            if (0 <= x < image.GetSize()[0]) and \
                (0 <= y < image.GetSize()[1]) and \
                (0 <= z < image.GetSize()[2]):
                # Get pixel value from the original image at the given coordinates
                pixel_value = image[x, y, z]
                blank_image[x, y, z] = pixel_value
                print(f"Pixel value at coordinates {coordinates}: {pixel_value}")
                output_images[region_name] = blank_image

    return output_images

image = data.get_3d_image("scan1")

def generate_regions():
    region1 = [[x, y, z] for x in range(0, 51) for y in range(0, 51) for z in range(0, 51)]
    region2 = [[x, y, z] for x in range(50, 101) for y in range(50, 101) for z in range(50, 101)]

    region_dict = {
        "Region1": region1,
        "Region2": region2
    }

    return region_dict


# Define your regions and their coordinates here
region_dict = generate_regions()

region_images = create_image_from_regions(image, region_dict)

# Display each region
for region_name, region_image in region_images.items():
    if region_image.GetNumberOfPixels() > 0:
        plt.figure(figsize=(6, 6))
        array_from_image = sitk.GetArrayFromImage(region_image)
        if array_from_image.any():
            # Displaying the first slice of the 3D image
            plt.imshow(array_from_image[0, :, :], cmap='gray')
            plt.axis('off')
            plt.title(f"Region: {region_name}")
            plt.show()
        else:
            print(f"Region: {region_name} is all black.")




