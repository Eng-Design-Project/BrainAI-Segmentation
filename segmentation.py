import SimpleITK as sitk

# Replace 'image.dcm' with the path to your DICOM file
image = sitk.ReadImage('mytest.dcm')

# Print the size of the image
print(image.GetSize())

