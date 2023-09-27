import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import skimage
import os
import pydicom
from skimage.transform import resize
import subprocess
import sys
from PIL import Image
#from pydicom import dcmread


def get_3d_png_array(directory):
    # Get a list of all PNG files in the directory
    scan_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".png")]
    
    # Sort files to ensure they are loaded in the correct order
    scan_files.sort()
    
    # Initialize an empty list to hold the 2D image data
    image_list = []
    
    # Loop through each file
    for file in scan_files:
        # Read the 2D image data and append it to the list
        image_data = np.asarray(Image.open(file))
        image_list.append(image_data)
    
    # Convert the list of 2D arrays into a 3D array
    volume_data = np.stack(image_list, axis=2)
    
    return volume_data

def get_3d_image(directory):
    # Get a list of all DICOM files in the directory
    scan_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".dcm")]
    # Read in the image series
    image = sitk.ReadImage(scan_files)
    return image

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

def display_seg_images(image_dict):
    for region, sitk_image in image_dict.items():
        view_sitk_3d_image(sitk_image, 5, region)

#note: simple ITK does not get all metadata, only most useful metadata for registration
def view_slice_metadata_from_directory(directory):
    scan_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".dcm")]
    for filename in scan_files:
        image = sitk.ReadImage(filename)
        print(image.GetMetaData("0020|0032"))

#takes all the dcm files in a directory, and returns a list of numpy pixel arrays of dimensions 224x224
#note, image registration (for the atlas segmentation) cannot be done
#  after an image has been converted to an array: the meta data used by sitk is lost
def resize_and_convert_to_3d_image(directory):
    image=get_3d_image(directory)
    array = sitk.GetArrayFromImage(image)
    new_images = []
    for i in range(0, array.shape[0]):
        new_images.append(resize(array[i,:,:], (224, 224), anti_aliasing=True))
    return new_images

#takes all of the dcm files in a directory, and saves them as png files in (string)new_dir
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
#save_dcm_dir_to_png_dir("scan4", "scan4_pngs")



#takes a directory and index, spits out filepath
def get_filepath(directory, index):
    filenames = [f for f in os.listdir(directory) if f.endswith(".dcm")]
    if index < len(filenames):
        return os.path.join(directory, filenames[index])
    else:
        return None

#takes sitk image and saves to directory as dcm files
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

        # Set the metadata attributes
        # slice_image.SetMetaData('0020|000D', 'registered scans') # Study Instance UID
        # slice_image.SetMetaData('0020|000E', 'registered scan') # Series Instance UID
        # slice_image.SetMetaData('0020|0011', str(z))    # Series Number
        # slice_image.SetMetaData('0020|0013', str(z))    # Instance Number

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

#temporarily "scan1" until atlas is complete
#hard coded, we only have 1 atlas
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

def rescale_image_test(orig_dir):
    orig_img = get_3d_image(orig_dir)
    new_img = rescale_image(orig_img)
    save_sitk_3d_img_to_dcm(new_img, "rescaled test")

#rescale_image_test("registered")

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

def test_store_seg_img_on_file(new_dir):
    ## the following code tests the "store_sec_img_on_file()"" functions
    directory1 = "scan1"
    directory2 = "scan2"
    image1 = get_3d_image(directory1)
    image2 = get_3d_image(directory2)
    dictionary = {"neocortex":image1, "frontal lobe":image2}
    store_seg_img_on_file(dictionary, new_dir)


# first argument should be a higher level folder with brain region subfolders containing DCM files.
# the output is a dictionary with brain region names as keys and sitk images as values
def subfolders_to_dictionary(directory):
    #print(os.listdir(directory))
    region_dict = {}
    for i in os.listdir(directory):
        # print(i)
        region_dict[i] = get_3d_image(os.path.join(directory, i))

    return region_dict

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
    
# function copied from segmentation
def DCMs_to_sitk_img_dict(directory):
    image = get_3d_image(directory)
    def generate_regions(): 
        region1 = [[x, y, z] for x in range(0, 51) for y in range(0, 51) for z in range(0, 51)]
        region2 = [[x, y, z] for x in range(50, 101) for y in range(50, 101) for z in range(0, 50)]

        region_dict = {
            "Region1": region1,
            "Region2": region2
        }
        return region_dict
    region_dict = generate_regions()
    region_images = create_seg_images(image, region_dict)
    #display_regions_from_dict(region_images)
    display_seg_images(region_images)
    return region_images

# the following code tests the "subfolders_to_dictionary()" function
def test_subfolders_to_dictionary(directory):
    regions = subfolders_to_dictionary(directory)
    for key, value in regions.items():
        view_sitk_3d_image(value, 15, key)
#test_store_seg_img_on_file("brain1")
#test_subfolders_to_dictionary("brain1")

# function copied from segmentation
def create_black_copy(image: sitk.Image) -> sitk.Image:
    # Create a copy of the input image
    black_image = sitk.Image(image.GetSize(), image.GetPixelID())
    black_image.SetOrigin(image.GetOrigin())
    black_image.SetSpacing(image.GetSpacing())
    black_image.SetDirection(image.GetDirection())

    # All pixel values are already set to 0 (black) upon initialization
    return black_image

#global variable
segmentation_results= None

# this function sets the global variable segmentation_results to a dictionary of regions:sitk images
# It takes an optional argument of a directory of DCMS. If no directory is passed, it uses "scan1"
def set_seg_results(directory = "scan1"):
    global segmentation_results
    segmentation_results = DCMs_to_sitk_img_dict(directory)
    print("segmentation results: ",segmentation_results.keys())

# set_seg_results()

   
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