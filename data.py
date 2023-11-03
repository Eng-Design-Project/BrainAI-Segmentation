import matplotlib.pyplot as plt
import numpy as np
#import SimpleITK as sitk
import skimage
from skimage.transform import resize
import os
import subprocess
import scipy
import sys
from PIL import Image
import pydicom
from pydicom import Dataset
from pydicom import dcmwrite
from pydicom import dcmread


#tried to use to load color atlas, to hard to parse coords
def get_3d_png_array(directory):
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

    # Stack all the 2D arrays into a single 3D array
    image_3d_array = np.stack(image_array_list, axis=-1)
    return image_3d_array

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
    image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))] # gather files
    slices = [pydicom.dcmread(os.path.join(directory, f)) for f in image_files] # read each file
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2])) # sorting and maintaining correct order
    return np.stack([s.pixel_array for s in slices])

    
# SITK TO PYDICOM - MD
# original:
'''
def view_sitk_3d_image(image, numSlices, displayText):
    array = sitk.GetArrayFromImage(image)

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
'''
def view_3d_image(image, numSlices, displayText):

    # Calculate the step size
    step = image.shape[0] // numSlices
    
    # Generate the slices
    slices = [image[i*step, :, :] for i in range(numSlices)]

    #display the slices
    fig, axes = plt.subplots(1, numSlices, figsize=(18, 18))

    # Set the title for the plot
    fig.suptitle(displayText, fontsize=16)

    for i, slice in enumerate(slices):
        axes[i].imshow(slice, cmap='gray')
        axes[i].axis('off')
    plt.show()

# SITK TO PYDICOM - MD
# original:
'''
def display_seg_images(image_dict):
    for region, sitk_image in image_dict.items():
        view_sitk_3d_image(sitk_image, 5, region)
'''
def display_seg_images(image_dict):
    for region, image in image_dict.items():
        view_3d_image(image, 5, region)

# SITK TO PYDICOM - MD
# original:
'''
def view_slice_metadata_from_directory(directory):
    scan_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".dcm")]
    for filename in scan_files:
        image = sitk.ReadImage(filename)
        print(image.GetMetaData("0020|0032"))
'''
#note: simple ITK does not get all metadata, only most useful metadata for registration
def view_slice_metadata_from_directory(directory):
    scan_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".dcm")]
    for filename in scan_files:
        image = pydicom.dcmread(filename)
        print(image.ImagePositionPatient)
        
    # imagepositionpatient doesn't always work very well so here's the same function, but going by file name instead of metadata:
'''
def view_filenames_from_directory(directory):
    scan_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".dcm")])
    for filename in scan_files:
        print(filename)
'''
        
# SITK TO PYDICOM - MD
# original:
'''
def resize_and_convert_to_3d_image(directory):
    image=get_3d_image(directory)
    array = sitk.GetArrayFromImage(image)
    new_images = []
    for i in range(0, array.shape[0]):
        new_images.append(resize(array[i,:,:], (224, 224), anti_aliasing=True))
    return new_images
'''
#takes all the dcm files in a directory, and returns a list of numpy pixel arrays of dimensions 224x224
#note, image registration (for the atlas segmentation) cannot be done
#  after an image has been converted to an array: the meta data used by sitk is lost
def resize_and_convert_to_3d_image(directory):
    image=get_3d_image(directory)
    array = image
    new_images = []
    for i in range(0, image.shape[0]):
        new_images.append(resize(array[i,:,:], (224, 224), anti_aliasing=True))
    return new_images
    
# SITK TO PYDICOM - MD
# original:
'''
def save_dcm_dir_to_png_dir(directory, new_dir):
    #create a directory called new_dir
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    # Get a list of all DICOM files in the directory
    scan_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".dcm")]
    # convert each file to a PNG and save it to the directory
    for i in range(0, len(scan_files)):
        image = sitk.ReadImage(scan_files[i])
        png_file = sitk.GetArrayFromImage(image)[0,:,:]
        output_file = os.path.basename(scan_files[i]).split(".")[0] + ".png"
        output_file_path = os.path.join(new_dir, output_file)
        plt.imsave(output_file_path, png_file, cmap='gray')
'''
#takes all of the dcm files in a directory, and saves them as png files in (string)new_dir
def save_dcm_dir_to_png_dir(directory, new_dir):
    #create a directory called new_dir
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    # Get a list of all DICOM files in the directory
    scan_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".dcm")]
    # convert each file to a PNG and save it to the directory
    for i in range(0, len(scan_files)):
        image = pydicom.dcmread(scan_files[i])
        png_file = image.pixel_array
        output_file = os.path.basename(scan_files[i]).split(".")[0] + ".png"
        output_file_path = os.path.join(new_dir, output_file)
        plt.imsave(output_file_path, png_file, cmap='gray')
#save_dcm_dir_to_png_dir("scan4", "scan4_pngs")



#takes a directory and index, spits out filepath
def get_filepath(directory, index):
    filenames = [f for f in os.listdir(directory) if f.endswith(".dcm")]
    if index < len(filenames):
        return os.path.join(directory, filenames[index])
    else:
        return None


# SITK TO PYDICOM - MD
# i'm not too confident about this one, please review.
# original:
'''
def save_sitk_3d_img_to_dcm(image, new_dir):
    # Check if the directory exists, if not, create it
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # Get the 3D image size to iterate through the slices
    size = image.GetSize()

    # Create a DICOM writer
    writer = sitk.ImageFileWriter()

    #study ID and series ID, hard coded for now, maybe based on user input later on

    # Iterate through the slices and save each one
    for z in range(size[2]):
        slice_image = image[:,:,z]
        slice_image = sitk.Cast(slice_image, sitk.sitkInt32)

        # Create a filename for the slice
        filename = os.path.join(new_dir, "slice_{:03d}.dcm".format(z))

        # Set the filename to the writer
        writer.SetFileName(filename)

        # Write the slice
        writer.Execute(slice_image)

        # Copy meta data to new slice
        atlas_dir = get_atlas_path()
        original_path = get_filepath(atlas_dir, z)
        #copy_meta_data(original_path, filename)

        print("Saved slice {} to {}".format(z, filename))

    print("Saved 3D image to {}".format(new_dir))
'''
#takes sitk image and saves to directory as dcm files
def save_sitk_3d_img_to_dcm(array, template_path, new_dir):
    # Check if the directory exists, if not, create it
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # Iterate through the slices and save each one
    for z in range(array.shape[0]):
        slice_arr = array[z,:,:]

        template = pydicom.dcmread(template_path)
        ds = Dataset(template)

        ds.PixelData = slice_arr.tobytes()
        ds.Rows, ds.Columns = slice_arr.shape
        ds.SliceLocation = str(z) 
        ds.InstanceNumber = str(z)

        filename = os.path.join(new_dir, "slice_{:03d}.dcm".format(z))
        dcmwrite(filename, ds)

        print("Saved slice {} to {}".format(z, filename))

    print("Saved 3D image to {}".format(new_dir))

#note, this function may have issues -Kevin
from pydicom.pixel_data_handlers.util import apply_modality_lut

def save_dicom_3d_img_to_png(dicom_dataset, new_dir):
    # Check if the directory exists, if not, create it
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # CHANGED: Get the 3D image array directly from pydicom Dataset
    image_array = dicom_dataset.pixel_array
    
    # Apply Modality LUT if present to get the correct HU or grayscale values
    image_array = apply_modality_lut(image_array, dicom_dataset)

    # CHANGED: Using shape attribute of the numpy array to get the dimensions
    depth, height, width = image_array.shape

    # CHANGED: Iterate through the slices using numpy slicing and save each one as PNG
    for z in range(depth):
        slice_image_np = image_array[z, :, :]
        slice_image_np = np.interp(slice_image_np, (slice_image_np.min(), slice_image_np.max()), (0, 255))
        slice_image_np = np.uint8(slice_image_np)

        # Create a filename for the slice
        filename = os.path.join(new_dir, "slice_{:03d}.png".format(z))

        # Save the slice as PNG using PIL (Python Imaging Library)
        slice_png = Image.fromarray(slice_image_np)
        slice_png.save(filename)

        print("Saved slice {} to {}".format(z, filename))

    print("Saved 3D image slices as PNG in {}".format(new_dir))

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

# SITK TO PYDICOM - MD
# original:
'''
def rescale_image(input_image):
    """
    Rescale the input image to have a size of 128x128 in width and height
    while keeping the depth the same.
    """
    # Original spacing and size
    original_spacing = input_image.GetSpacing()
    original_size = input_image.GetSize()

    # New size (keeping the depth the same)
    new_size = [128, 128, original_size[2]]
    
    # Compute new spacing given the original and new sizes
    new_spacing = [
        original_spacing[0] * (original_size[0] / new_size[0]),
        original_spacing[1] * (original_size[1] / new_size[1]),
        original_spacing[2]
    ]

    # Use the Resample function to rescale the image
    resampled_image = sitk.Resample(input_image, new_size, sitk.Transform(), 
                                    sitk.sitkLinear, input_image.GetOrigin(),
                                    new_spacing, input_image.GetDirection(), 0.0,
                                    input_image.GetPixelID())

    return resampled_image
'''
def rescale_image(input_image_np):
    """
    Rescale the input image to have a size of 128x128 in width and height
    while keeping the depth the same.
    """
    original_size = input_image_np.shape

    # New size (keeping the depth the same)
    new_size = [original_size[2], 128, 128]

    resampled_image_np = scipy.ndimage.zoom(input_image_np, (original_size[0]/new_size[0], original_size[1]/new_size[1], 1))

    return resampled_image_np

# SITK TO PYDICOM - MD
# original:
'''
def rescale_image_test(orig_dir):
    orig_img = get_3d_image(orig_dir)
    new_img = rescale_image(orig_img)
    save_sitk_3d_img_to_dcm(new_img, "rescaled test")
'''
def rescale_image_test(orig_dir):
    orig_img_np = get_3d_image(orig_dir)
    new_img_np = rescale_image(orig_img_np)
    save_sitk_3d_img_to_dcm(new_img_np, "rescaled test")

#rescale_image_test("registered")

# SITK TO PYDICOM - MD
# original:
'''
# this function takes a dictionary as input - with the keys being brain region names and 
# the values being sitk images, then converts the sitk image to dcm and stores it in
# a subfolder based on the key (brain region name) which in turn is stored in a higher
# level folder (new_dir)
def store_seg_img_on_file(dict, new_dir):
    # Check if the directory exists, if not, create it (higher level folder)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    for key in dict:
        # making a sub folder based on the brain region name
        sub_dir = os.path.join(new_dir, key)
        os.makedirs(sub_dir)

        save_sitk_3d_img_to_dcm(dict[key], sub_dir)
        #print("key:", key)
'''

def store_seg_img_on_file(img_dict, new_dir):
    # Check if the directory exists, if not, create it (higher level folder)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    for key in img_dict:
        # making a sub folder based on the brain region name
        sub_dir = os.path.join(new_dir, key)
        os.makedirs(sub_dir)

        save_sitk_3d_img_to_dcm(img_dict[key], sub_dir)

# SITK TO PYDICOM - MD
# original:
'''
def test_store_seg_img_on_file(new_dir):
    ## the following code tests the "store_sec_img_on_file()"" functions
    directory1 = "scan1"
    directory2 = "scan2"
    image1 = get_3d_image(directory1)
    image2 = get_3d_image(directory2)
    dictionary = {"neocortex":image1, "frontal lobe":image2}
    store_seg_img_on_file(dictionary, new_dir)
'''
def test_store_seg_img_on_file(new_dir):
    ## the following code tests the "store_sec_img_on_file()"" functions
    directory1 = "scan1"
    directory2 = "scan2"
    image1 = get_3d_image(directory1)
    image2 = get_3d_image(directory2)
    img_dict = {"neocortex":image1, "frontal lobe":image2}
    store_seg_img_on_file(img_dict, new_dir)


# SITK TO PYDICOM - MD
# original:
'''
#note, this function may have issues, I haven't tested it exetensively -Kevin
def store_seg_png_on_file(dict, new_dir):
    # Check if the directory exists, if not, create it (higher level folder)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    for key in dict:
        # making a sub folder based on the brain region name
        sub_dir = os.path.join(new_dir, key)
        os.makedirs(sub_dir)

        save_sitk_3d_img_to_png(dict[key], sub_dir)
        #print("key:", key)
'''

#note, this function may have issues, I haven't tested it exetensively -Kevin
def store_seg_png_on_file(img_dict, new_dir):
    # Check if the directory exists, if not, create it (higher level folder)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    for key in img_dict:
        # making a sub folder based on the brain region name
        sub_dir = os.path.join(new_dir, key)
        os.makedirs(sub_dir)

        save_dicom_3d_img_to_png(img_dict[key], sub_dir)
        #print("key:", key)

# SITK TO PYDICOM - MD
# original:
'''
# the function below takes a dictionary of sitk images and returns an equivalent dict of png images
# img_dict parameter is a dictionary whose keys are strings, and values are sITK 3d images
def sitk_dict_to_png_dict(img_dict):
    png_dict = {} # this will be the dict of PNGs
    for key in img_dict:
        # Create a new nested dictionary for the keya
        png_dict[key] = {}
        # Get the 3D image size to iterate through the slices
        size = img_dict[key].GetSize()
        # Iterate through the slices and save each one as PNG
        for z in range(size[2]):
            slice_image = img_dict[key][:,:,z]
            slice_image_np = sitk.GetArrayFromImage(slice_image)
            slice_image_np = np.interp(slice_image_np, (slice_image_np.min(), slice_image_np.max()), (0, 255))
            slice_image_np = np.uint8(slice_image_np)

            slice_png = Image.fromarray(slice_image_np)
            png_dict[key][z] = slice_png
    #print("PNG DICTIONARY HAS BEEN GENERATED")       
    return png_dict
'''
def sitk_dict_to_png_dict(img_dict):
    png_dict = {} # this will be the dict of PNGs
    for key in img_dict:
        # Create a new nested dictionary for the keya
        png_dict[key] = {}
        full_image_np = img_dict[key]

        for z in range(full_image_np.shape[0]):
            slice_image_np = full_image_np[z,:,:]
            slice_image_np = np.interp(slice_image_np, (slice_image_np.min(), slice_image_np.max()), (0, 255))
            slice_image_np = np.uint8(slice_image_np)

            slice_png = Image.fromarray(slice_image_np)
            png_dict[key][z] = slice_png

    #print("PNG DICTIONARY HAS BEEN GENERATED")       
    return png_dict

def convert_sitk_dict_to_numpy(sitk_dict):
    numpy_dict = {}
    for key, image in sitk_dict.items():
        if isinstance(image, sitk.Image):
            numpy_array = sitk.GetArrayFromImage(image)
            numpy_dict[key] = numpy_array
        else:
            raise ValueError(f"Value for key '{key}' is not a SimpleITK image.")

    return numpy_dict


# first argument should be a higher level folder with brain region subfolders containing DCM files.
# the output is a dictionary with brain region names as keys and sitk images as values
'''def subfolders_to_dictionary(directory):
    #print(os.listdir(directory))
    region_dict = {}
    for i in os.listdir(directory):
        # print(i)
        region_dict[i] = get_3d_image(os.path.join(directory, i))

    return region_dict'''
def subfolders_to_dictionary(directory):
    region_dict = {}
    for i in os.listdir(directory):
        region_dict[i] = get_3d_image(os.path.join(directory, i))
    return region_dict

# SITK TO PYDICOM - MD
# original:
'''
    # function copied from segmentation 
def create_seg_images(image, region_dict):
    output_images = {}
    for region_name, coordinates_list in region_dict.items():
        blank_image = create_black_copy(image)
        
        for coordinates in coordinates_list:
            x, y, z = coordinates
            if (0 <= x < image.GetSize()[0]) and \
               (0 <= y < image.GetSize()[1]) and \
               (0 <= z < image.GetSize()[2]):
                pixel_value = image[x, y, z]
                blank_image[x, y, z] = pixel_value
                
        # Append the finished blank_image to the output_images dictionary
        output_images[region_name] = blank_image
    #print(f"Size of output images:  {len(output_images)}")
    return output_images
'''
def create_seg_images(image_np, region_dict):
    output_images_np = {}
    for region_name, coordinates_list in region_dict.items():
        blank_image_np = np.zeros_like(image_np)
        
        for coordinates in coordinates_list:
            x, y, z = coordinates
            if (0 <= x < image_np.shape[1]) and \
               (0 <= y < image_np.shape[0]) and \
               (0 <= z < image_np.shape[2]):
                pixel_value = image_np[y, x, z]
                blank_image_np[y, x, z] = pixel_value
                
        # Append the finished blank_image to the output_images dictionary
        output_images_np[region_name] = blank_image_np
    #print(f"Size of output images:  {len(output_images)}")
    return output_images_np
    

#test_store_seg_img_on_file("brain1")
#test_subfolders_to_dictionary("brain1")

# SITK TO PYDICOM - MD
# i had chatgpt help with this one, not sure if it even makes sense tbh. please review
# original:
'''
# function copied from segmentation
def create_black_copy(image: sitk.Image) -> sitk.Image:
    # Create a copy of the input image
    black_image = sitk.Image(image.GetSize(), image.GetPixelID())
    black_image.SetOrigin(image.GetOrigin())
    black_image.SetSpacing(image.GetSpacing())
    black_image.SetDirection(image.GetDirection())

    # All pixel values are already set to 0 (black) upon initialization
    return black_image
'''
def create_black_copy(image_np):
    return np.zeros_like(image_np)

#global variable
segmentation_results= None

# this function sets the global variable segmentation_results to a dictionary of regions:sitk images
# It takes an optional argument of a directory of DCMS. If no directory is passed, it uses "scan1"
def set_seg_results(directory = "atl_segmentation_DCMs"):
    global segmentation_results

    segmentation_results = subfolders_to_dictionary(directory)
    print("segmentation results: ",segmentation_results.keys())
    #note: the function DCMs_to_etc, is a dummy function that grabs a single scan from memory and then 
    # splits it into a dict. We don't need this at all. We should assume this function is 
    # passed the segment results dict, and then it removes the skull with del seg_results["Skull"]

# set_seg_results()

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


#this is an alternate way to get np arrays from dcm folder,
#  may be better/worse, unsure until other one gives us an error
'''
def get_3d_image(directory):
    # Get a list of all DICOM files in the directory
    scan_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".dcm")])
    slices = [ ]
    for s in scan_files:
        dicom_slice = pydicom.dcmread(s)
        numpy_slice = dicom_slice.pixel_array

        if ('RescaleSlope' in dicom_slice) and ('RescaleIntercept' in dicom_slice):
            slope = float(dicom_slice.RescaleSlope)
            intercept = float(dicom_slice.RescaleIntercept)
            numpy_slice = slope * numpy_slice + intercept

        slices.append(numpy_slice)
    
    image = np.stack(slices, axis=0) 
    return image
'''