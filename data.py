import matplotlib.pyplot as plt
import numpy as np
import skimage
import os
import pydicom
from pydicom.dataset import Dataset
from skimage.transform import resize
from scipy.ndimage import zoom
import subprocess
import sys
from PIL import Image
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


# folder of DCM images as input
def get_3d_image(directory):
    image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]  # gather files
    slices = [pydicom.dcmread(os.path.join(directory, f)) for f in image_files]  # read each file
    
    # Sort slices, but ensure the float conversion is safely handled
    slices.sort(key=lambda x: float(getattr(x, 'ImagePositionPatient', [0,0,0])[2]))
    
    # Reverse the list so that the stack is in the opposite order
    slices = slices[::-1]
    
    return np.stack([s.pixel_array for s in slices])
        
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
def get_atlas_path():
    atlas_dir = "atlas"
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
    region_dict = {}
    for i in os.listdir(directory):
        # print(i)
        region_dict[i] = get_3d_image(os.path.join(directory, i))

    return region_dict

    

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

    # Normalize each image to the range [0, 255]
    normalized_images = (images - np.min(images)) / (np.max(images) - np.min(images)) * 255

    # Calculate the average pixel brightness for all normalized images
    average_brightness = np.mean(normalized_images, axis=(1, 2))
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


if __name__ == "__main__":
    test_dir = "scan1"
    test_pydicom_arr = get_3d_image(test_dir)
    display_3d_array_slices(test_pydicom_arr, 20)
    
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