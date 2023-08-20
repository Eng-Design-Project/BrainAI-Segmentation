import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import skimage
import os
import pydicom
from skimage.transform import resize
#from pydicom import dcmread


def copy_meta_data(sourcefile_dcm, targetfile_dcm):
    # source dcm file
    source_dcm = pydicom.dcmread(sourcefile_dcm)

    # read new file
    target_dcm = pydicom.dcmread(targetfile_dcm)

    '''
    # Copy specific metadata elements from source
    target_dcm.PatientName = source_dcm.PatientName
    target_dcm.PatientID = source_dcm.PatientID
    # ... etc
    '''
    #copy all elements from source
    target_dcm.update(source_dcm)

    target_dcm.save_as(targetfile_dcm)

def new_func(targetfile_dcm):
    return targetfile_dcm

def save_segmented_image(segmented_image, new_directory):
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    # 3D image size iterates through slices
    size = segmented_image.GetSize()

    # DICOM writer
    dicom_writer = sitk.ImageFileWriter()

    # Iterate through the slices and save each one
    for z in range(size[2]):
        slice_image = segmented_image[:,:,z]
        slice_image = sitk.Cast(slice_image, sitk.sitkInt32)

        # name the file
        filename = os.path.join(new_directory, "segmented_slice_{:03d}.dcm".format(z))

        # assign file name to writer
        dicom_writer.SetFileName(filename)

        # Write the slice
        dicom_writer.Execute(slice_image)

        # Copy metadata from the original DCM file, through atlas image
        #will need to redefine this once we have the atlas directory and created its path
        atlas_directory = get_atlas_path()
        original_path = get_filepath(atlas_directory, z)
        copy_meta_data(original_path, filename)

        print("Saved segmented slice {} to {}".format(z, filename))

    print("Saved segmented image to {}".format(new_directory))


def get_3d_image(directory):
    # Get a list of all DICOM files in the directory
    scan_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".dcm")]
    # Read in the image series
    image = sitk.ReadImage(scan_files)
    return image

def view_sitk_3d_image(image, numSlices):
    array = sitk.GetArrayFromImage(image)

    # Calculate the step size
    step = array.shape[0] // numSlices
    
    # Generate the slices
    slices = [array[i*step, :, :] for i in range(numSlices)]

    #display the slices
    fig, axes = plt.subplots(1, numSlices, figsize=(18, 18))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice, cmap='gray')
        axes[i].axis('off')
    plt.show()

def view_slice_metadata_from_directory(directory):
    scan_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".dcm")]
    for filename in scan_files:
        image = sitk.ReadImage(filename)
        print(image.GetMetaData("0020|0032"))

# Path to the directory that contains the DICOM files
directory1 = "scan1"
directory2 = "scan2"

# Create 3d image with SITK
image1 = get_3d_image(directory1)
image2 = get_3d_image(directory2)

#view slices of 3d image
view_sitk_3d_image(image1, 10)
view_sitk_3d_image(image2, 10)

#view metadata of slices in directory
view_slice_metadata_from_directory(directory1)
view_slice_metadata_from_directory(directory2)


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