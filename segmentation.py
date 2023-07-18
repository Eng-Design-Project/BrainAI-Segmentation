import SimpleITK as sitk

# Replace 'image.dcm' with the path to your DICOM file
image = sitk.ReadImage('mytest.dcm')

# Print the size of the image
print(image.GetSize())

# def segment_images(*images):
#this will recieve a tuple of unspecified size as arguments, 
#a loop will go through and call segment_image on each

def segment_image(image):
    #register image to unmarked atlas
    #create blank image (stack of 2d arrays) for each region
    #each region has a set of 3d coordinates, created from the atlas
    #for each region, at the atlas coordinates
    #   pixel value = registered image pixel value at same coordinate
    #   other pixels are blank, black, or some other special value to indicate the segment boundaries
    #return set of images, one image for each region

    #a more efficient method:
    #   create a single blank image (3d array)
    #   each entry is a tuple of two values:
    #   the pixel value of the registered image at the same coordinate
    #   a number to indicate the region of that pixel
    #pros: more compact, can classify coordinates with guarunteed no overlap, 
    #   DL and Clustering can use enter image as data rather than seperate segments
    #cons: have to make custom function to view the segmentations, 
    #   DL and Clustering more complex?

    print("function incomplete")