import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import numpy as np
import skimage
import os
import pydicom
from pydicom.dataset import Dataset
from skimage.transform import resize, rescale
from scipy.ndimage import zoom
import subprocess
import sys
from PIL import Image
from skimage.feature import canny
from skimage.color import rgb2gray
#from pydicom import dcmread

#tried to use to load color atlas, to hard to parse coords
# def get_3d_png_array(directory):
#     image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".png")]
#     image_array_list = []

#     for image_file in image_files:
#         # Open the image using Pillow
#         img = Image.open(image_file)
#         # Convert the Pillow image to a numpy array
#         img_arr = np.array(img)
#         # Check if the array is already 128x128x3
#         if img_arr.shape == (128, 128, 3):
#             # If it is, use it as is
#             image_array_list.append(img_arr)
#         else:
#             # If it's not, convert the image to RGB
#             rgb_img = img.convert('RGB')
#             # Convert the RGB image to a numpy array and append to the list
#             image_array_list.append(np.array(rgb_img))

#     # Stack all the 2D arrays into a single 3D array
#     image_3d_array = np.stack(image_array_list, axis=-1)
#     return image_3d_array

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

def display_3d_array_slices(img_3d, num_slices):
    print(img_3d.shape)
    # Ensure that num_slices does not exceed the number of available slices
    num_slices = min(num_slices, img_3d.shape[0])

    # Calculate indices for evenly spaced slices
    indices = np.linspace(0, img_3d.shape[0] - 1, num_slices).astype(int)

    # Calculate grid dimensions
    cols = int(np.ceil(np.sqrt(num_slices)))
    rows = int(np.ceil(num_slices / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    
    for i, index in enumerate(indices):
        ax = axes.flat[i]
        ax.imshow(img_3d[index, :, :], cmap='gray')
        ax.axis('off')  # Hide the axis
        ax.set_title(f'Slice {index}')

    # Hide any remaining empty subplots
    for i in range(num_slices, rows * cols):
        axes.flat[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def save_2d_images_list(image_list, directory):
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for i, img_array in enumerate(image_list):
        # Convert the numpy array to a Pillow Image object
        img = Image.fromarray(img_array)
        
        # Construct a file name for each image
        file_name = f'image_{i + 1:03d}.png'
        
        # Create the full path to the file
        file_path = os.path.join(directory, file_name)
        
        # Save the image
        img.save(file_path)

''' old def get_3d_image(directory):
# 
#     image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))] # gather files
#     slices = [pydicom.dcmread(os.path.join(directory, f)) for f in image_files] # read each file
#     slices.sort(key=lambda x: float(x.ImagePositionPatient[2])) # sorting and maintaining correct order
#     return np.stack([s.pixel_array for s in slices])

# folder of DCM images as input
# def get_3d_image(directory):
#     image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]  # gather files
#     slices = [pydicom.dcmread(os.path.join(directory, f)) for f in image_files]  # read each file

#     # Create a set to track unique ImagePositionPatient[2] values
#     unique_positions = set()
#     unique_slices = []

#     # Loop through each slice and add it to unique_slices if its position is unique
#     for s in slices:
#         position = float(s.ImagePositionPatient[2])
#         if position not in unique_positions:
#             unique_slices.append(s)
#             unique_positions.add(position)

#     # Sort the unique slices
#     unique_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

#     return np.stack([s.pixel_array for s in unique_slices])
'''


def get_3d_image(directory):
    print("Reading DICOM files from directory:", directory)

    image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    print(f"Found {len(image_files)} files in the directory.")

    slices = [pydicom.dcmread(os.path.join(directory, f)) for f in image_files]
    print(f"Read {len(slices)} DICOM slices.")

    unique_positions = set()
    unique_slices = []

    # Loop through each slice
    for s in slices:
        position = float(s.ImagePositionPatient[2])
        if position not in unique_positions:
            unique_slices.append(s)
            unique_positions.add(position)

    print(f"Number of unique slices based on position: {len(unique_slices)}")

    # Sort the unique slices
    unique_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    print("Unique slices sorted based on ImagePositionPatient[2].")

    # Stack and reverse the array
    output_arr = np.stack([s.pixel_array for s in unique_slices])
    output_arr = output_arr[::-1, :, :]

    print("Final output array shape:", output_arr.shape)
    return output_arr


        
# def get_3d_image(directory):
#     image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))] # gather files
#     slices = [pydicom.dcmread(os.path.join(directory, f)) for f in image_files] # read each file
#     slices.sort(key=lambda x: float(x.ImagePositionPatient[2])) # sorting and maintaining correct order
#     return np.stack([s.pixel_array for s in slices])

def view_np_3d_image(np_array_image, numSlices, displayText):
    array = np_array_image
    # Calculate the step size
    step = array.shape[0] // numSlices
    # Generate the slices
    slices = [array[i*step, :, :] for i in range(numSlices)]
    #display the slices
    fig, axes = plt.subplots(1, numSlices, figsize=(18, 18))
    # Set the title for the plot
    fig.suptitle(displayText, fontsize=16)
    for i, slice in enumerate(slices):
        axes[i].imshow(slice, cmap='gray')
        axes[i].axis('off')
    plt.show()


def display_seg_np_images(image_dict):
    for region, np_image in image_dict.items():
        view_np_3d_image(np_image, 5, region)

def view_metadata_from_directory(directory):
    # List all files in the given directory that have a .dcm extension
    scan_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(".dcm")]
    
    for filename in scan_files:
        # Read the DICOM file
        dicom_data = pydicom.dcmread(filename)
        
        # Print all the available metadata
        print(f"Metadata for {filename}:")
        for metadata in dicom_data.dir():
            try:
                value = getattr(dicom_data, metadata)
                print(f"{metadata}: {value}")
            except AttributeError:
                # In case the metadata is not present in the DICOM file, we skip it
                continue
        
        # Add an empty line for better readability between different files
        print("\n")

#takes all the dcm files in a directory, and returns a list of numpy pixel arrays of dimensions 224x224
#note, image registration (for the atlas segmentation) cannot be done
#function not used anywhere, and returns a list of slices. Keep it?
def resize_and_convert_to_3d_image(directory):
    array=get_3d_image(directory)
    new_images = []
    for i in range(0, array.shape[0]):
        new_images.append(resize(array[i,:,:], (224, 224), anti_aliasing=True))
    return new_images

#takes all of the dcm files in a directory, and saves them as png files in (string)new_dir
def save_dcm_dir_to_png_dir(directory, new_dir):
    # Create a directory called new_dir if it doesn't exist
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # Assuming get_3d_image returns a 3D NumPy array from all DICOM files in the directory
    image_3d = get_3d_image(directory)
    
    # Iterate over each slice in the 3D image array
    for i, image_slice in enumerate(image_3d):
        # Convert the slice to an 8-bit grayscale image
        image_slice_normalized = (np.clip(image_slice, 0, np.max(image_slice)) / np.max(image_slice) * 255).astype(np.uint8)
        # Create a PIL image from the NumPy array
        img = Image.fromarray(image_slice_normalized)
        # Construct the output filename for each slice
        slice_filename = f"slice_{i:03}.png"
        img.save(os.path.join(new_dir, slice_filename))



#takes a directory and index, spits out filepath
def get_filepath(directory, index):
    filenames = [f for f in os.listdir(directory) if f.endswith(".dcm")]
    if index < len(filenames):
        return os.path.join(directory, filenames[index])
    else:
        return None

#given a directory, it will return the first path to a dcm file.
#  Good for save_3d_img_to_dcm, makes it easier to use
def get_first_dcm_path(directory):
    # List all DICOM files in the directory
    dcm_files = [f for f in os.listdir(directory) if f.endswith('.dcm')]
    if not dcm_files:
        raise FileNotFoundError("No DICOM files found in the directory.")

    # Return the full path of the first DICOM file
    return os.path.join(directory, dcm_files[0])

#takes image and saves to directory as dcm files
def save_3d_img_to_dcm(array, template_dir, new_dir):
    # Check if the directory exists, if not, create it
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # Generate template_path from template_dir
    template_path = get_first_dcm_path(template_dir)

    # Read the template DICOM file to get the slice thickness
    template = pydicom.dcmread(template_path)
    slice_thickness = float(template.SliceThickness)

    # Iterate through the slices and save each one
    for z in range(array.shape[0]):
        slice_arr = array[z, :, :]
        
        # Create a new DICOM dataset based on the template
        ds = Dataset()
        ds.file_meta = template.file_meta
        ds.update(template)  # Update the dataset with the template

        # Update the pixel data
        ds.PixelData = slice_arr.tobytes()

        # Update the rows and columns based on the array shape
        ds.Rows, ds.Columns = slice_arr.shape

        # Calculate the slice position (ImagePositionPatient[2])
        initial_position = float(template.ImagePositionPatient[2])
        ds.ImagePositionPatient[2] = str(initial_position + z * slice_thickness)

        # Update the instance number
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


def save_3d_img_to_png(image_array, new_dir):
    # Check if the directory exists, if not, create it
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # Get the number of slices in the z dimension
    num_slices = image_array.shape[0]

    # Iterate through the slices and save each one as PNG
    for z in range(num_slices):
        slice_image_np = image_array[z, :, :]

        # Process the slice image
        processed_slice_image_np = bring_edges_to_boundary(slice_image_np)
        # Resize the slice
        pil_img = Image.fromarray(processed_slice_image_np)
        resized_img = pil_img.resize((224, 224), Image.Resampling.LANCZOS)
        resized_slice_image_np = np.array(resized_img)
        # Normalize the slice for better contrast in the PNG
        resized_slice_image_np = np.interp(resized_slice_image_np, (resized_slice_image_np.min(), resized_slice_image_np.max()), (0, 255))
        resized_slice_image_np = np.uint8(resized_slice_image_np)

        # Create a filename for the slice
        filename = os.path.join(new_dir, f"slice_{z:03d}.png")

        # Save the slice as PNG
        plt.imsave(filename, resized_slice_image_np, cmap='gray')

        print(f"Saved slice {z} to {filename}")

    print(f"Saved 3D image slices as PNG in {new_dir}")

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


#just spits out "atlas"
def get_atlas_path(size):

    if size < 50: 
        size = 0

    if size > 50 and size < 100:
        size = 1

    if size >100:
        return 0
    

    switch = {
        0 : ("atlas" , "color atlas"),
        1 : ("large atlas" , "color atlas large")
    }

    atlas_dir = switch[size]  #dictionary of different names
    print(atlas_dir[0], atlas_dir[1])
    return atlas_dir

def get_file_path():
    
    initial_dir = os.getcwd()  # Get the current working directory

    #folder_path = filedialog.askdirectory(initialdir=initial_dir, title="Select a folder")

    #return folder_path

#selected_folder = get_file_path()
#print("Selected folder:", selected_folder)



def open_folder_dialog():
    if sys.platform.startswith('win'):
        subprocess.run(['explorer', '/select,', '.'], shell=True)
    elif sys.platform.startswith('darwin'):
        subprocess.run(['open', '-a', 'Finder', '.'])
    elif sys.platform.startswith('linux'):
        subprocess.run(['xdg-open', '.'])
    else:
        print("Unsupported platform")

def rescale_image(input_array):
    """
    Rescale the input array to have a size of 128x128 in width and height
    while keeping the depth the same.
    """
    # Original size
    original_size = input_array.shape

    # New size (keeping the depth the same)
    new_size = [128, 128, original_size[2]]
    
    # Calculate the zoom factors for each dimension
    zoom_factors = [
        new_size[0] / original_size[0],
        new_size[1] / original_size[1],
        1  # keep depth the same
    ]

    # Use the zoom function to rescale the image
    resampled_array = zoom(input_array, zoom_factors, order=1)  # order=1 for bilinear interpolation

    return resampled_array

def rescale_image_test(orig_dir):
    orig_img = get_3d_image(orig_dir)
    new_img = rescale_image(orig_img)
    save_3d_img_to_dcm(new_img, "atlas", "rescaled test")

#rescale_image_test("registered")

# this function takes a dictionary as input - with the keys being brain region names and 
# the values being 3d np array images, then converts the 3d np array image to dcm and stores it in
# a subfolder based on the key (brain region name) which in turn is stored in a higher
# level folder (new_dir)
def store_seg_img_on_file(dict, template_dir, new_dir):
    # Check if the directory exists, if not, create it (higher level folder)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    for key in dict:
        # making a sub folder based on the brain region name
        if key != "Skull":
            sub_dir = os.path.join(new_dir, key)
            os.makedirs(sub_dir)

            save_3d_img_to_dcm(dict[key], template_dir, sub_dir)
            #print("key:", key)

def test_store_seg_img_on_file(new_dir):
    ## the following code tests the "store_sec_img_on_file()"" functions
    directory1 = "scan1"
    directory2 = "scan2"
    image1 = get_3d_image(directory1)
    image2 = get_3d_image(directory2)
    dictionary = {"neocortex":image1, "frontal lobe":image2}
    store_seg_img_on_file(dictionary, "scan1", new_dir)

#note, this function may have issues, I haven't tested it exetensively -Kevin
def store_seg_png_on_file(dict, new_dir):
    # Check if the directory exists, if not, create it (higher level folder)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    for key in dict:
        if key != "Skull":
            # making a sub folder based on the brain region name
            sub_dir = os.path.join(new_dir, key)
            os.makedirs(sub_dir)

            save_3d_img_to_png(dict[key], sub_dir)
            #print("key:", key)

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



# first argument should be a higher level folder with brain region subfolders containing DCM files.
# the output is a dictionary with brain region names as keys and 3d np array images as values
def subfolders_to_dictionary(directory):
    #print(os.listdir(directory))
    is_valid = is_segment_results_dir(directory)
    if (is_valid):
        region_dict = {}
        for i in os.listdir(directory):
            # print(i)
            print(os.path.join(directory, i))
            region_dict[i] = get_3d_image(os.path.join(directory, i))
            print("shape of " + i)
            print(region_dict[i].shape)
        
        return region_dict
    else:
        return None
    

# the following code tests the "subfolders_to_dictionary()" function
def test_subfolders_to_dictionary(directory):
    regions = subfolders_to_dictionary(directory)
    for key, value in regions.items():
        view_np_3d_image(value, 15, key)
#test_store_seg_img_on_file("brain1")
#test_subfolders_to_dictionary("brain1")


#global variable
segmentation_results= None

# this function sets the global variable segmentation_results to a dictionary of regions:3d np arrays
# It takes an optional argument of a directory of DCMS. If no directory is passed, it uses "atl_segmentation_DCMs"
def set_seg_results_with_dir(directory = "atl_segmentation_DCMs"):
    global segmentation_results

    segmentation_results = subfolders_to_dictionary(directory)
    print("segmentation results: ",segmentation_results.keys())
    
# set_seg_results()

#this function needs fixing, there should be a for loop like in the function below it.
#this function has been changed, it now returns a single NORMALIZED average.
#this function takes a 3d numpy image, returns a single number as the average of (46) averages
def average_overall_brightness_3d(image):
    # Ensure the input is a numpy array
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    # Normalize each image to the range [0, 255]
    normalized_images = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255

    # Calculate the average pixel brightness for all normalized images
    normalized_image_averages = np.mean(normalized_images, axis=(1, 2))

    # Calculate the overall average by averaging the image averages
    overall_average_brightness = np.mean(normalized_image_averages)
    return overall_average_brightness

# added another fix to this function.
# this fuction has been changed, it now returns NORMALIZED (between [0, 255]) averages
# if the argument is a 3d image with 46 slices, this function will return a numpy array with 46 averages
# each average can be accessed by typical indexing, e.g. average_brightness[0] returns first average, ...[45] returns last, etc.
def array_of_average_pixel_brightness_3d(images):
    # Ensure the input is a numpy array
    if not isinstance(images, np.ndarray):
        raise ValueError("Input must be a numpy array")
    # Ensure the input is a 3D array
    if len(images.shape) != 3:
        raise ValueError("Input must be a 3D array of grayscale images")

    num_slices, height, width = images.shape

    # Initialize an array to store the normalized average brightness for each image
    average_brightness = np.zeros(num_slices)

    for i in range(num_slices):
        # Check if the range is zero to avoid division by zero
        if np.max(images[i]) - np.min(images[i]) == 0:
            normalized_image = images[i]  # Avoid normalization if the range is zero
        else:
            # Normalize each image to the range [0, 255]
            normalized_image = (images[i] - np.min(images[i])) / (np.max(images[i]) - np.min(images[i])) * 255

        # Calculate the average pixel brightness for the normalized image
        average_brightness[i] = np.mean(normalized_image)

    return average_brightness

# this function returns the normalized [0,255] avg brightness of a single 2d grayscale numpy image
def avg_brightness_2d(image):
    # Ensure the input image is a numpy array
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a numpy array")
    # Ensure the image is 2D (grayscale)
    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale (2D array)")
    # Calculate the average pixel brightness

    # Normalize image values to the range [0, 255]
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255

    # Calculate the average pixel brightness of the normalized image
    average_brightness = np.mean(normalized_image)
    return average_brightness

# img_dict is a dictionary of 3d arrays, and coords dict is a dict of lists of coordinates
# they both have matching regions, then it goes through the regions and finds the brighntess of a pixel
# for every coordinate in the list of coordinates, normalizes it, calculates the average of all of them,
# and appends to a dict key = region, value = average brightness
# and returns average brightness dictionary
def avg_brightness(img_dict, coords_dict):
    # Ensure the input is a numpy array
    brightness_dict = {}

    for region in img_dict.keys():
        if not isinstance(img_dict[region], np.ndarray):
            raise ValueError("Input must be a numpy array")

        avg_brightness = 0
        max = np.max(img_dict[region])
        min = np.min(img_dict[region])
        print("MIN:")
        print(min)
        print("MAX:")
        print(max)
        if max - min == 0:
            continue
        else:
            print("LENGTH:")
            print(len(coords_dict[region]))
            print('LAST ELEMENT')
            print(coords_dict[region][len(coords_dict[region])-1])
            print(img_dict[region].shape[0])
            print(img_dict[region].shape[1])
            print(img_dict[region].shape[2])
            count = 0
            normalized_image = min_max_normalize(img_dict[region])
            for i in range(len(coords_dict[region])):
                x, y, z = coords_dict[region][i]
                if (0 <= x < img_dict[region].shape[0]) and \
                    (0 <= y < img_dict[region].shape[1]) and \
                    (0 <= z < img_dict[region].shape[2]):
                    count+=1
                    pixel_value = normalized_image[z, y, x]
                    # Normalize each image to the range [0, 255]
                    # pixel_value = ((pixel_value - min) / (max - min)) * 255
                    # Calculate the average pixel brightness for the normalized image
                    if z > 44:
                        print(pixel_value)
                    avg_brightness += pixel_value

            print('AVERAGE BRIGHTNESS')
            print(avg_brightness)
            if count != 0:
                avg_brightness = avg_brightness / count
        brightness_dict[region] = avg_brightness * 255

    return brightness_dict

def min_max_normalize(arr):
    arr64 = arr.astype(np.float64)

    min_val = np.min(arr64)
    max_val = np.max(arr64)

    # Check if max and min values are the same (to avoid division by zero)
    if max_val - min_val == 0:
        return arr64

    normalized_arr64 = (arr64 - min_val) / (max_val - min_val)
    return normalized_arr64


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

def save_3d_img_to_jpg(image_array, new_dir):
    # Check if the directory exists, if not, create it
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # Get the number of slices in the z dimension
    num_slices = image_array.shape[0]

    # Iterate through the slices and save each one as JPG
    for z in range(num_slices):
        slice_image_np = image_array[z, :, :]

        # Process the slice image with bring_edges_to_boundary function
        processed_slice_image_np = bring_edges_to_boundary(slice_image_np)  # Assuming this function is defined elsewhere

        # Resize the slice
        pil_img = Image.fromarray(processed_slice_image_np)
        resized_img = pil_img.resize((224, 224), Image.Resampling.LANCZOS)
        resized_slice_image_np = np.array(resized_img)

        # Normalize the slice for better contrast in the JPEG
        resized_slice_image_np = np.interp(resized_slice_image_np, (resized_slice_image_np.min(), resized_slice_image_np.max()), (0, 255))
        resized_slice_image_np = np.uint8(resized_slice_image_np)

        # Create a filename for the slice
        filename = os.path.join(new_dir, f"slice_{z:03d}.jpg")

        # Save the slice as JPEG
        plt.imsave(filename, resized_slice_image_np, cmap='gray')

        print(f"Saved slice {z} to {filename}")

    print(f"Saved 3D image slices as JPEG in {new_dir}")

def convert_png_to_jpg(source_dir, target_dir, quality=95):
    """
    Convert all PNG images in a directory to grayscale JPG format.

    Args:
    source_dir (str): Directory containing PNG images.
    target_dir (str): Directory where JPG images will be saved.
    quality (int): Quality of JPG images, between 0 and 95.
    """
    # Process each file in the source directory
    for file_name in os.listdir(source_dir):
        if file_name.endswith(".png"):
            # Construct full file path
            file_path = os.path.join(source_dir, file_name)

            # Open the image
            with Image.open(file_path) as img:
                # Convert the image to Grayscale
                grayscale_img = img.convert('L')

                # Construct the output file path
                jpg_file_name = os.path.splitext(file_name)[0] + '.jpg'
                jpg_file_path = os.path.join(target_dir, jpg_file_name)

                # Save the image in JPG format with specified quality
                grayscale_img.save(jpg_file_path, 'JPEG', quality=quality)

                print(f"Converted '{file_name}' to '{jpg_file_name}'")
            
"""def crop_image_to_boundary(image, border=5):
    
    Crop an image so that any edge is within 5 pixels of the image boundary.

    Args:
    image (PIL.Image): The image to be cropped.
    border (int): The distance from the edge to stop cropping.

    Returns:
    PIL.Image: The cropped image.
    
    # Convert image to numpy array
    np_image = np.array(image)

    # Find the coordinates of the first non-black pixels from each edge
    rows = np.any(np_image > 0, axis=1)
    cols = np.any(np_image > 0, axis=0)

    # Check if the image contains non-black pixels
    if not np.any(rows) or not np.any(cols):
        # If the image is entirely black, return the original image
        return image

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # Adjust for the border
    ymin = max(ymin - border, 0)
    ymax = min(ymax + border, image.height)
    xmin = max(xmin - border, 0)
    xmax = min(xmax + border, image.width)

    # Crop and return the image
    cropped_image = image.crop((xmin, ymin, xmax, ymax))
    return cropped_image"""    

def bring_edges_to_boundary(image_array, boundary=5, target_size=(224, 224)):
    # Determine if the image is grayscale or color
    is_color = len(image_array.shape) == 3

    # Convert to grayscale for edge detection
    gray_image = rgb2gray(image_array) if is_color else image_array

    # Detect edges
    edges = canny(gray_image)

    # Find the coordinates of the detected edges
    rows, cols = np.where(edges)

    if rows.size > 0 and cols.size > 0:
        # Crop the image to bring edges to the desired boundary
        min_row, max_row = max(np.min(rows) - boundary, 0), min(np.max(rows) + boundary, image_array.shape[0])
        min_col, max_col = max(np.min(cols) - boundary, 0), min(np.max(cols) + boundary, image_array.shape[1])
        cropped_image = image_array[min_row:max_row, min_col:max_col]

        # Calculate the aspect ratio of the cropped image
        height, width = cropped_image.shape[:2]
        aspect_ratio = height / width

        # Calculate new size maintaining aspect ratio
        if height > width:
            new_height = target_size[0]
            new_width = round(new_height / aspect_ratio)
        else:
            new_width = target_size[1]
            new_height = round(new_width * aspect_ratio)

        # Resize the image maintaining the aspect ratio
        resized_image = resize(cropped_image, (new_height, new_width), anti_aliasing=True)

        # Calculate padding to reach target size
        pad_height = target_size[0] - new_height
        pad_width = target_size[1] - new_width

        # Define padding for color and grayscale images
        if is_color:
            pad_width_tuple = ((pad_height//2, pad_height - pad_height//2), (pad_width//2, pad_width - pad_width//2), (0, 0))
        else:
            pad_width_tuple = ((pad_height//2, pad_height - pad_height//2), (pad_width//2, pad_width - pad_width//2))

        # Pad the resized image to match target size
        final_image = np.pad(resized_image, pad_width_tuple, mode='constant', constant_values=0)
    else:
        # If no edges are found, use the original image
        final_image = resize(image_array, target_size, anti_aliasing=True)

    return final_image

def crop_3d_grayscale_image(image):
    """
    Crops a 3D grayscale image (4D array with single color channel) to remove surrounding black voxels.

    Args:
    image (np.array): The 4D grayscale image array to be cropped.

    Returns:
    np.array: Cropped image array.
    """

    # Find indices where the voxel is not black (0)
    non_black_indices = np.where(image[:, :, :, 0] != 0)

    # Find min and max indices along each axis
    min_x, max_x = non_black_indices[0].min(), non_black_indices[0].max()
    min_y, max_y = non_black_indices[1].min(), non_black_indices[1].max()
    min_z, max_z = non_black_indices[2].min(), non_black_indices[2].max()

    # Crop the image
    cropped_image = image[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1, :]

    return cropped_image 


"""def bring_edges_to_boundary(image_array, boundary=5):
    # Determine if the image is grayscale or color
    is_color = len(image_array.shape) == 3

    # Convert to grayscale for edge detection
    gray_image = rgb2gray(image_array) if is_color else image_array

    # Detect edges
    edges = canny(gray_image)

    # Find the coordinates of the detected edges
    rows, cols = np.where(edges)

    # Check if rows and cols are not empty
    if rows.size > 0 and cols.size > 0:
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)

        # Calculate new boundaries considering the desired boundary space
        min_row = max(min_row - boundary, 0)
        min_col = max(min_col - boundary, 0)
        max_row = min(max_row + boundary, image_array.shape[0])
        max_col = min(max_col + boundary, image_array.shape[1])

        # Crop the image to bring edges to the desired boundary
        cropped_image = image_array[min_row:max_row, min_col:max_col]

        # Define padding for color and grayscale images
        if is_color:
            pad_width = ((boundary, boundary), (boundary, boundary), (0, 0))
        else:
            pad_width = ((boundary, boundary), (boundary, boundary))

        # Pad the cropped image
        padded_image = np.pad(cropped_image, pad_width, mode='constant', constant_values=0)
    else:
        # If no edges are found, use the original image
        padded_image = image_array

    # Resize if necessary to fit into 224x224
    final_image = resize(padded_image, (224, 224), anti_aliasing=True)

    return final_image""" 
    
       

"""def store_seg_jpg_on_file(input_dir, temp_dir, output_dir):
    
  #  Processes images in a directory by zooming in on edges and then converting them to JPEGs.
   # :param input_dir: Directory with input PNG images.
    #:param temp_dir: Temporary directory to store processed images before conversion.
   # :param output_dir: Directory to store the final JPEG images.

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            # Read the image
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            img_array = np.array(img)

            # Apply 'bring_edges_to_boundary'
            processed_img_array = bring_edges_to_boundary(img_array)

            # Convert numpy array back to an Image object
            processed_img = Image.fromarray(processed_img_array)

            # Save the processed image temporarily
            temp_img_path = os.path.join(temp_dir, filename)
            processed_img.save(temp_img_path)

    # Convert the processed images to JPEGs
    convert_pngs_to_jpegs(temp_dir, output_dir)""" 

def pad_to_aspect_ratio(image, target_size, background_color=0):
    """
    Pad an image to a given aspect ratio.

    Args:
    image (PIL.Image): The image to pad.
    target_size (tuple): The target width and height.
    background_color (int): The color to use for padding (grayscale).

    Returns:
    PIL.Image: Padded Image.
    """
    width, height = image.size
    target_width, target_height = target_size

    # Calculate the new size, maintaining the aspect ratio
    aspect_ratio = width / height
    target_aspect_ratio = target_width / target_height

    if aspect_ratio > target_aspect_ratio:
        # Image is wider than desired aspect ratio
        new_height = int(width / target_aspect_ratio)
        offset = (new_height - height) // 2
        padded = ImageOps.expand(image, (0, offset, 0, offset), fill=background_color)
    else:
        # Image is taller than desired aspect ratio
        new_width = int(height * target_aspect_ratio)
        offset = (new_width - width) // 2
        padded = ImageOps.expand(image, (offset, 0, offset, 0), fill=background_color)

    return padded.resize(target_size, Image.LANCZOS)           

  
def store_seg_jpg_on_file(dict, new_dir):
    # Check if the directory exists, if not, create it (higher level folder)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    for key in dict:
        if key != "Skull":
            # making a sub folder based on the brain region name
            sub_dir = os.path.join(new_dir, key)
            os.makedirs(sub_dir)

            save_3d_img_to_jpg(dict[key], sub_dir)
            #print("key:", key)
# def store_seg_jpg_on_file(seg_dict, new_dir):
#     """
#     Store segmentation results as JPEG images in specified directory.

#     :param seg_dict: Dictionary with keys as region names and values as 3D numpy arrays.
#     :param template_dir: Directory containing DICOM files for metadata.
#     :param new_dir: Directory where JPEG images will be saved.
#     """
#     # Ensure the new directory exists
#     if not os.path.exists(new_dir):
#         os.makedirs(new_dir)

#     # Iterate through each segmented region in the dictionary
#     for region_name, seg_array in seg_dict.items():
#         # Create a subdirectory for each region
#         region_dir = os.path.join(new_dir, region_name)
#         if not os.path.exists(region_dir):
#             os.makedirs(region_dir)

#         # Iterate through each slice of the 3D array
#         for i, slice_array in enumerate(seg_array):
#             # Normalize and convert slice to uint8
#             slice_array_normalized = (np.clip(slice_array, 0, np.max(slice_array)) / np.max(slice_array) * 255).astype(np.uint8)

#             # Create a PIL image from the numpy array
#             slice_image = Image.fromarray(slice_array_normalized)

#             # Pad and resize the image
#             resized_image = pad_to_aspect_ratio(slice_image, (224, 224))

#             # Construct the filename for the slice
#             slice_filename = os.path.join(region_dir, f"{region_name}_slice_{i:03d}.jpg")
#             resized_image.save(slice_filename, format='JPEG', quality=100)

#             print(f"Saved {slice_filename} in {region_dir}")



if __name__ == "__main__":
    
    test_dir = "scan1"
    test_pydicom_arr = get_3d_image(test_dir)
    save_3d_img_to_jpg(test_pydicom_arr, "test_jpg_scan1")

    # display_3d_array_slices(test_pydicom_arr, 20)
    
    # print(is_segment_results_dir("atl_segmentation_DCMs"))
    # print(is_segment_results_dir("atl_segmentation_PNGs"))
    # print(is_segment_results_dir("atlas"))
    # print(is_segment_results_dir("atlas_pngs"))
    # print(contains_only_dcms("atlas"))

# Path to the directory that contains the DICOM files
#directory1 = "scan1"
#directory2 = "scan2"

# Create 3d image with SITK
#image1 = get_3d_image(directory1)
#image2 = get_3d_image(directory2)

#view slices of 3d image
#view_sitk_3d_image(image1, 10, "scan1")
#view_sitk_3d_image(image2, 10, "scan2")

#view metadata of slices in directory
#view_slice_metadata_from_directory(directory1)
#view_slice_metadata_from_directory(directory2)

#Save all of the DCM files in directory1 as PNG files in new directory
##save_sitk_3d_img_png(directory1, "PNGs")

#get the current working directory
#current_directory = os.getcwd()
#contstruct the file path
#file_path = os.path.join(current_directory, 'mytest.dcm')


#pixarray = ds.pixel_array
#plt.imshow(pixarray, 'gray', origin='lower')
# plt.show()

#print("scan 1 size: ",len(ds.pixel_array))
#print("image dimenstions before: ", pixarray.shape) #this prints the dimensions, e.g. (128, 128)

#the code below resizes the image to (224, 224)
#resized_img=resize(pixarray, (224, 224), anti_aliasing=True)
#plt.imshow(resized_img, 'gray', origin='lower')
# plt.show()
#print("image dimensions after: ",resized_img.shape)


#Save the resized image in PNG format
'''
output_dir = ""
output_file = os.path.join(output_dir, file_path.split(".")[0]+".png")
plt.imsave(output_file, resized_img)
'''
# Below is an example of a tuple of dcm files, the tuple can be used as a parameter for
# the function below it
'''
dcm_files = (
    os.path.join(current_directory, 'scan1.dcm'),
    os.path.join(current_directory, 'scan2.dcm'),
    os.path.join(current_directory, 'scan3.dcm')
)
def get_dcm_files():
    dcm_files = (
        os.path.join(current_directory, 'scan1.dcm'),
        os.path.join(current_directory, 'scan2.dcm'),
        os.path.join(current_directory, 'scan3.dcm')
    )
    return dcm_files
'''
# the function below converts a tuple of DCM files to a list of pixel arrays and displays it with plt
# it takes a tuple as its parameter, and returns a list
'''
def p_arrays(*files):
    output_list = []
    for i in files:
        d = dcmread(i)
        p_array = d.pixel_array
        img_resize= resize(p_array, (224,224), anti_aliasing=True)
        output_list.append(img_resize)
    return output_list
'''

#p_array_list = p_arrays(*dcm_files)
# "p_array_list" is now a list of pixel arrays

# below is an example of accessing a pixel array from the list, and displaying it with plt
#plt.imshow(p_array_list[2], 'gray', origin='lower')
#plt.show()

'''
#won't work can't apply pixeldata from one dcm to another
#pixeldata is dependent on metadata
def copy_metadata(metadata_dcm_file, target_dcm_file):
    # Read the original DICOM file
    metadata_dcm = pydicom.dcmread(metadata_dcm_file)

    # Read the target DICOM file (the one you want to copy the metadata to)
    target_dcm = pydicom.dcmread(target_dcm_file)

    pixel_data = target_dcm.PixelData

    metadata_dcm.PixelData = pixel_data

    # Copy metadata from original to target
    # for elem in original_dcm.iterall():
    #     if elem.tag != (0x7FE0, 0x0010) and elem.tag.group not in (0x0002, 0x0004):
    #         target_dcm.add(elem)

    # Save the target DICOM file with the updated metadata
    metadata_dcm.save_as(target_dcm_file)
#copy_meta_data('scan1', 'registered')

#doesn't work because pixel data is dependent on metadata
# can't apply pixel data from one dcm to another, it just doesn't work 
def full_copy_metadata(metadata_dcm_dir, target_dcm_dir):
    
    metadata_files = [os.path.join(metadata_dcm_dir, f) for f in os.listdir(metadata_dcm_dir) if f.endswith(".dcm")]
    target_files = [os.path.join(target_dcm_dir, f) for f in os.listdir(target_dcm_dir) if f.endswith(".dcm")]
    #final_files = [os.path.join(target_dcm_dir, f) for f in os.listdir(metadata_dcm_dir) if f.endswith(".dcm")]

    
    for i in range(min(len(metadata_files), len(target_files))):
        metadata_img = pydicom.dcmread(metadata_files[i])
        target_img = pydicom.dcmread(target_files[i])
        metadata_img.PixelData = target_img.PixelData
        
        metadata_img.save_as(target_files[i])
#full_copy_metadata("scipy_reg_image_dcm", "scipy_reg_image_dcm")
'''