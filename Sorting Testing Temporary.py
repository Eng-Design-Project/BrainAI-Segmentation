# import os

# def reverse_file_order(folder_path, file_extension):
#     # Find all files with the specified extension
#     files = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]

#     # Extract the numbers from the filenames and sort them
#     filenumbers = [int(f.split('_')[-1].split('.')[0]) for f in files]
#     filenumbers.sort()

#     # Find the maximum number to set the new naming scheme
#     max_number = max(filenumbers)

#     # First, rename all files to a temporary naming scheme
#     temp_files = []
#     for f in files:
#         temp_filename = f.replace(file_extension, '.temp' + file_extension)
#         os.rename(os.path.join(folder_path, f), os.path.join(folder_path, temp_filename))
#         temp_files.append(temp_filename)

#     # Rename the temporary files to the new names
#     for temp_f in temp_files:
#         old_number = int(temp_f.split('_')[-1].split('.')[0])
#         new_number = max_number - old_number
#         new_filename = temp_f.replace('.temp', '').replace(str(old_number), str(new_number))
#         os.rename(os.path.join(folder_path, temp_f), os.path.join(folder_path, new_filename))


# folder_path = "Sorted_Brain_Scans_PNG_Colored - Copy"
# file_extension = ".png"  # or ".dcm" for DCM files
# reverse_file_order(folder_path, file_extension)

# import os
# import pydicom
# from PIL import Image

# def convert_png_to_jpeg(source_folder, target_folder):
#     # Create target folder if it doesn't exist
#     if not os.path.exists(target_folder):
#         os.makedirs(target_folder)

#     # Iterate over all dcm files in the source folder
#     for filename in os.listdir(source_folder):
#         if filename.endswith('.png'):
#             # Construct full file path
#             dcm_path = os.path.join(source_folder, filename)
#             jpeg_path = os.path.join(target_folder, filename.replace('.png', '.jpeg'))

#             # Read the DICOM file
#             dcm_data = pydicom.read_file(dcm_path)

#             # Extract pixel array from DICOM file
#             image = dcm_data.pixel_array

#             # Convert to PIL image (assuming the image is grayscale)
#             pil_image = Image.fromarray(image).convert('L')

#             # Save as JPEG
#             pil_image.save(jpeg_path)

# # Paths
# source_folder = r'C:\Users\Justin Rivera\OneDrive\Documents\ED1\BrainAI-Segmentation\atlas_pngs' # Replace with your DICOM folder path
# target_folder = os.path.join(source_folder, '../JPEG_Conversions') # Change as needed

# convert_png_to_jpeg(source_folder, target_folder)

import os
import numpy as np
import pydicom
from PIL import Image

def normalize_image(image):
    image = image.astype(float)
    min_val = np.min(image)
    max_val = np.max(image)
    if min_val != max_val:  # Avoid division by zero
        image = ((image - min_val) / (max_val - min_val)) * 255.0
    return image.astype(np.uint8)

def convert_dcm_to_jpg(source_folder, target_folder):
    # Create target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Iterate over all dcm files in the source folder
    for filename in os.listdir(source_folder):
        if filename.endswith('.dcm'):
            # Construct full file path
            dcm_path = os.path.join(source_folder, filename)
            jpg_path = os.path.join(target_folder, filename.replace('.dcm', '.jpg'))

            # Read the DICOM file
            dcm_data = pydicom.read_file(dcm_path)

            # Extract pixel array from DICOM file
            image = dcm_data.pixel_array

            # Normalize the image to 0-255
            normalized_image = normalize_image(image)

            # Convert to PIL image
            pil_image = Image.fromarray(normalized_image)

            # Save as JPG
            pil_image.save(jpg_path)

# Paths
source_folder = r'C:\Users\Justin Rivera\OneDrive\Documents\ED1\BrainAI-Segmentation\atl_segmentation_DCMs\Brain' # Replace with your DICOM folder path
target_folder = os.path.join(source_folder, '../JPEG_Conversions 2') # Change as needed

convert_dcm_to_jpg(source_folder, target_folder)