# you need to pip install the following:
# pip install numpy
# pip install matplotlib
# pip install pydicom
# pip install scikit.image

import matplotlib.pyplot as plt
import numpy as np
import skimage
import os

from skimage.transform import resize
from pydicom import dcmread

#get the current working directory
current_directory = os.getcwd()
#contstruct the file path
file_path = os.path.join(current_directory, 'mytest.dcm')

ds = dcmread(file_path)
#print(ds.PixelData)

pixarray = ds.pixel_array
plt.imshow(pixarray, 'gray', origin='lower')
# plt.show()

#print("scan 1 size: ",len(ds.pixel_array))
print("image dimenstions before: ", pixarray.shape) #this prints the dimensions, e.g. (128, 128)

#the code below resizes the image to (224, 224)
resized_img=resize(pixarray, (224, 224), anti_aliasing=True)
plt.imshow(resized_img, 'gray', origin='lower')
# plt.show()
print("image dimensions after: ",resized_img.shape)


#Save the resized image in PNG format
output_dir = ""
output_file = os.path.join(output_dir, file_path.split(".")[0]+".png")
plt.imsave(output_file, resized_img)


# task: making a method that the project core will call to open up a foler and grab a scan
# the function below converts a single DCM file to a pixel array and displays it with plt
def open_file(file_path):
    d = dcmread(file_path)
    p_array = d.pixel_array
    img_resize= resize(p_array, (224, 224), anti_aliasing=True) #resizes image to 224x224
    #plt.imshow(img_resize, 'gray', origin='lower') #(displaying the pixel array is optional)
    #plt.show()

# Below is an example of a tuple of dcm files, the tuple can be used as a parameter for
# the function below it
dcm_files = (
    os.path.join(current_directory, 'scan1.dcm'),
    os.path.join(current_directory, 'scan2.dcm'),
    os.path.join(current_directory, 'scan3.dcm')
)


# the function below converts a tuple of DCM files to a list of pixel arrays and displays it with plt
# it takes a tuple as its parameter, and returns a list
def p_arrays(*files):
    output_list = []
    for i in files:
        d = dcmread(i)
        p_array = d.pixel_array
        img_resize= resize(p_array, (224,224), anti_aliasing=True)
        output_list.append(img_resize)
    return output_list


p_array_list = p_arrays(*dcm_files)
# "p_array_list" is now a list of pixel arrays

# below is an example of accessing a pixel array from the list, and displaying it with plt
plt.imshow(p_array_list[2], 'gray', origin='lower')
plt.show()