import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import skimage
import os
import pydicom
from skimage.transform import resize
#from pydicom import dcmread



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

#note: simple ITK does not get all metadata, only most useful metadata for registration
def view_slice_metadata_from_directory(directory):
    scan_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".dcm")]
    for filename in scan_files:
        image = sitk.ReadImage(filename)
        print(image.GetMetaData("0020|0032"))

#takes all the dcm files in a directory, and returns a list of numpy pixel arrays of dimensions 224x224
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

def copy_meta_data(metadata_dcm_file, target_dcm_file):
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

def get_atlas_path():
    atlas_dir = "scan1"
    return atlas_dir
def get_file_path():
    
    initial_dir = os.getcwd()  # Get the current working directory

    #folder_path = filedialog.askdirectory(initialdir=initial_dir, title="Select a folder")

    #return folder_path

#selected_folder = get_file_path()
#print("Selected folder:", selected_folder)

import subprocess
import sys

def open_folder_dialog():
    if sys.platform.startswith('win'):
        subprocess.run(['explorer', '/select,', '.'], shell=True)
    elif sys.platform.startswith('darwin'):
        subprocess.run(['open', '-a', 'Finder', '.'])
    elif sys.platform.startswith('linux'):
        subprocess.run(['xdg-open', '.'])
    else:
        print("Unsupported platform")

#open_folder_dialog()    


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