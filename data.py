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
plt.show()

#print("scan 1 size: ",len(ds.pixel_array))
print("image dimenstions before: ", pixarray.shape) #this prints the dimensions, e.g. (128, 128)

#the code below resizes the image to (224, 224)
resized_img=resize(pixarray, (224, 224), anti_aliasing=True)
plt.imshow(resized_img, 'gray', origin='lower')
plt.show()
print("image dimensions after: ",resized_img.shape)