#use get 3d image (sitk)
#use the numpy version
# testing the following functions from data: view_np_3d_image(), display_seg_np_images(), and convert_sitk_dict_to_numpy()

import data
import SimpleITK as sitk
import os

# first argument should be a higher level folder with brain region subfolders containing DCM files.
# the output is a dictionary with brain region names as keys and np arrays(images) as values
# note: this function isn't useful because I run into problems when coverting segmented DCMs to numpy arrays
def subfolders_to_np_dictionary(directory):
    #print(os.listdir(directory))
    region_dict = {}
    for i in os.listdir(directory):
        # print(i)
        region_dict[i] = data.get_3d_array_from_file(os.path.join(directory, i))

    return region_dict


#Testing functions which display sitk images
image1 = data.get_3d_image("scan1")
image1 = sitk.GetArrayFromImage(image1)
#data.display_3d_array_slices(image1, 10)

#Testing functions to display numpy 3d images (works)
image2 = data.get_3d_array_from_file("scan1")
#data.display_3d_array_slices(image2, 10)


# now I want to get sitk dict, display it, then numpy dict and display it
sitk_dict = data.subfolders_to_dictionary("atl_segmentation_DCMs")
# this should display 5 slices
data.display_seg_images(sitk_dict)


# statement below doesn't work, I presume because the segmented DCMs lack the relevant metadata (we use pydicom.dcmread() to convert DCMs into np arrays)
    #"subfolders_to_np_dictionary" uses "get_3d_array_from_file" which is causing the error
# np_dict = subfolders_to_np_dictionary("atl_segmentation_DCMs")
# display_seg_np_images(np_dict)

#array = sitk.GetArrayFromImage(sitk_image) is used to convert sitk image to np array
# I want a function to convert a dict of regions:sitk images to a dict of regions:np_arrays
# I should find a function that takes an sitk image and turns it into a dictionary

np_dict = data.convert_sitk_dict_to_numpy(sitk_dict)
data.display_seg_np_images(np_dict)
#data.view_np_3d_image(np_dict["Brain"],10, "brain")
