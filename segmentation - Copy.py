#dicom_path2 = "ADNI_007_S_1339_PT_TRANSAXIAL_BRAIN_3D_FDG_ADNI_CTAC__br_raw_20070402142342176_8_S29177_I47688.dcm"
#dicom_path1 = "ADNI_007_S_1339_PT_TRANSAXIAL_BRAIN_3D_FDG_ADNI_CTAC__br_raw_20070402142342697_9_S29177_I47688.dcm"

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import data


# Path to the directory that contains the DICOM files
directory1 = "scan1"
directory2 = "scan2"

# Create 3d image with SITK
image1 = data.get_3d_image(directory1)
image2 = data.get_3d_image(directory2)
#does it need to by cast to float32?

#set up the registration framework
registration_method = sitk.ImageRegistrationMethod()

#set similarity metric
registration_method.SetMetricAsMeanSquares()

#set optimizer
registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
registration_method.SetOptimizerScalesFromPhysicalShift()

#initial transform
initial_transform = sitk.TranslationTransform(image1.GetDimension())
registration_method.SetInitialTransform(initial_transform)

#set interpolator
registration_method.SetInterpolator(sitk.sitkLinear)

#execute registration
final_transform = registration_method.Execute(sitk.Cast(image1, sitk.sitkFloat32), sitk.Cast(image2, sitk.sitkFloat32))

#apply transformation
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(image1)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(100)
resampler.SetTransform(final_transform)

registered_image = resampler.Execute(image2)

data.view_sitk_3d_image(image1, 10, "image1")
data.view_sitk_3d_image(image2, 10, "image2")
data.view_sitk_3d_image(registered_image, 10, "regImage")
