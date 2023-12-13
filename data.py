import matplotlib.pyplot as plt
import numpy as np
import os
from os import path
import pydicom
from pydicom.dataset import Dataset
from PIL import Image

# folder of DCM images as input
def get_3d_image(directory):
    image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]  # gather files
    slices = [pydicom.dcmread(os.path.join(directory, f)) for f in image_files]  # read each file
    
    # Sort slices, but ensure the float conversion is safely handled
    slices.sort(key=lambda x: float(getattr(x, 'ImagePositionPatient', [0,0,0])[2]))
    
    # Reverse the list so that the stack is in the opposite order
    slices = slices[::-1]
    
    return np.stack([s.pixel_array for s in slices])









### need to bring "atlas" dir and "color atlas dir" to this project
#38
#currently used for loading color atlas
def get_2d_png_array_list(directory):
    image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".png")]
    image_array_list = []

    for image_file in image_files:
        # Open the image using Pillow
        img = Image.open(image_file)
        # Convert the Pillow image to a numpy array
        img_arr = np.array(img)
        # Check if the array is already 128x128x3
        if img_arr.shape == (128, 128, 3):
            # If it is, use it as is
            image_array_list.append(img_arr)
        else:
            # If it's not, convert the image to RGB
            rgb_img = img.convert('RGB')
            # Convert the RGB image to a numpy array and append to the list
            image_array_list.append(np.array(rgb_img))

    return image_array_list

#205
#given a directory, it will return the first path to a dcm file.
#  Good for save_3d_img_to_dcm, makes it easier to use
def get_first_dcm_path(directory):
    # List all DICOM files in the directory
    dcm_files = [f for f in os.listdir(directory) if f.endswith('.dcm')]
    if not dcm_files:
        raise FileNotFoundError("No DICOM files found in the directory.")

    # Return the full path of the first DICOM file
    return os.path.join(directory, dcm_files[0])

#216
#takes image and saves to directory as dcm files
def save_3d_img_to_dcm(array, template_dir, new_dir):
    # Check if the directory exists, if not, create it
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # Generate template_path from template_dir
    template_path = get_first_dcm_path(template_dir)

    # Iterate through the slices and save each one
    for z in range(array.shape[0]):
        slice_arr = array[z, :, :]

        # Read the template DICOM file
        template = pydicom.dcmread(template_path)

        # Create a new DICOM dataset based on the template
        ds = Dataset()
        ds.file_meta = template.file_meta
        ds.update(template)  # Update the dataset with the template

        # Update the pixel data
        ds.PixelData = slice_arr.tobytes()

        # Update the rows and columns based on the array shape
        ds.Rows, ds.Columns = slice_arr.shape

        # Update the slice location and instance number
        ds.SliceLocation = str(z)
        ds.InstanceNumber = str(z)

        # Set the endianess and VR encoding
        ds.is_little_endian = True
        ds.is_implicit_VR = True

        # Create a filename for the slice
        filename = os.path.join(new_dir, f"slice_{z:03d}.dcm")

        # Save the dataset using pydicom
        pydicom.write_file(filename, ds, write_like_original=False)

        print(f"Saved slice {z} to {filename}")

    print(f"Saved 3D image to {new_dir}")

#261
def save_3d_img_to_png(image_array, new_dir):
    # Check if the directory exists, if not, create it
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # Get the number of slices in the z dimension
    num_slices = image_array.shape[0]

    # Iterate through the slices and save each one as PNG
    for z in range(num_slices):
        slice_image_np = image_array[z, :, :]
        # Normalize the slice for better contrast in the PNG
        slice_image_np = np.interp(slice_image_np, (slice_image_np.min(), slice_image_np.max()), (0, 255))
        slice_image_np = np.uint8(slice_image_np)

        # Create a filename for the slice
        filename = os.path.join(new_dir, f"slice_{z:03d}.png")

        # Save the slice as PNG
        plt.imsave(filename, slice_image_np, cmap='gray')

        print(f"Saved slice {z} to {filename}")

    print(f"Saved 3D image slices as PNG in {new_dir}")

#286
def convert_3d_numpy_to_png_list(np_3d):
    png_list = []
    length = len(np_3d[:][:])
    for i in reversed(range(length)):
        image = np_3d[:][:][i]
        image = np.interp(image, (image.min(), image.max()), (0, 255))
        image = np.uint8(image)
        png = Image.fromarray(image)
        png_list.append(png)
    return png_list

# 298
#just spits out "atlas"
def get_atlas_path():
    print("PATH:",path.abspath(path.join(path.dirname(__file__))))
    path_to_atlas = path.abspath(path.join(path.dirname(__file__), 'atlas'))
    print("NEW PATH:",path_to_atlas)
    atlas_dir = "atlas"
    return path_to_atlas

#just spits out "color atlas"
def get_color_atlas_path():
    print("PATH:",path.abspath(path.join(path.dirname(__file__))))
    path_to_atlas = path.abspath(path.join(path.dirname(__file__), 'color atlas'))
    print("NEW PATH:",path_to_atlas)
    return path_to_atlas


#360
def store_seg_img_on_file(dict, template_dir, new_dir):
    # Check if the directory exists, if not, create it (higher level folder)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    for key in dict:
        # making a sub folder based on the brain region name
        sub_dir = os.path.join(new_dir, key)
        os.makedirs(sub_dir)

        save_3d_img_to_dcm(dict[key], template_dir, sub_dir)
        #print("key:", key)

#383
def store_seg_png_on_file(dict, new_dir):
    # Check if the directory exists, if not, create it (higher level folder)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    for key in dict:
        # making a sub folder based on the brain region name
        sub_dir = os.path.join(new_dir, key)
        os.makedirs(sub_dir)

        save_3d_img_to_png(dict[key], sub_dir)
        #print("key:", key)

#396
# the function below takes a dictionary of 3d np array images and returns an equivalent dict of png images
# img_dict parameter is a dictionary whose keys are strings, and values are 3d np array 3d images
def array_dict_to_png_dict(img_dict):
    png_dict = {} # this will be the dict of PNGs
    for key in img_dict:
        # Create a new nested dictionary for the keya
        png_dict[key] = {}
        # Get the 3D image size to iterate through the slices
        size = img_dict[key].shape
        # Iterate through the slices and save each one as PNG
        for z in range(size[0]):
            slice_image_np = img_dict[key][z,:,:]
            slice_image_np = np.interp(slice_image_np, (slice_image_np.min(), slice_image_np.max()), (0, 255))
            slice_image_np = np.uint8(slice_image_np)

            slice_png = Image.fromarray(slice_image_np)
            png_dict[key][z] = slice_png
    #print("PNG DICTIONARY HAS BEEN GENERATED")       
    return png_dict

#418
# first argument should be a higher level folder with brain region subfolders containing DCM files.
# the output is a dictionary with brain region names as keys and 3d np array images as values
def subfolders_to_dictionary(directory):
    #print(os.listdir(directory))
    region_dict = {}
    for i in os.listdir(directory):
        # print(i)
        region_dict[i] = get_3d_image(os.path.join(directory, i))

    return region_dict

#440
#global variable
segmentation_results= None

# 494
def is_segment_results_dir(directory):
    """
    Validates if a given directory matches the specified structure and format.
    :param directory: The directory to validate.
    :return: True if the directory matches the format, False otherwise.
    """
    
    # Check if top-level directory contains only directories and no files
    if any(os.path.isfile(os.path.join(directory, entry)) for entry in os.listdir(directory)):
        print("input dir contains a file")
        return False
    
    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(directory):

        # In the top-level folder, we expect only directories (subfolders for each 3D image)
        if dirpath == directory:
            continue

        # For each subfolder (3D image set), it should contain only .dcm files and no subfolders
        for dirname in dirnames:
            print("dir found in subdir, should only be files")
            return False  # If we find any directory inside a subfolder, it's invalid
        
        for filename in filenames:
            if not filename.endswith('.dcm'):
                print("found non-dcm")
                return False

    return True

#526
def contains_only_dcms(directory):
    """
    Validates if a given directory contains only .dcm files and no subdirectories.
    :param directory: The directory to validate.
    :return: True if the directory contains only .dcm files, False otherwise.
    """
    
    for dirpath, dirnames, filenames in os.walk(directory):
        # If there are any subdirectories, return False
        if dirnames:
            return False
        
        # Check if all files have the .dcm extension
        for filename in filenames:
            if not filename.endswith('.dcm'):
                return False

    return True