#dicom_path2 = "ADNI_007_S_1339_PT_TRANSAXIAL_BRAIN_3D_FDG_ADNI_CTAC__br_raw_20070402142342176_8_S29177_I47688.dcm"
#dicom_path1 = "ADNI_007_S_1339_PT_TRANSAXIAL_BRAIN_3D_FDG_ADNI_CTAC__br_raw_20070402142342697_9_S29177_I47688.dcm"

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import data


def basic_segment(image1, image2, 
                  simMetric="MeanSquares", optimizer="GradientDescent", 
                  interpolator="Linear", samplerInterpolator="Linear"):
    

    #set up the registration framework
    registration_method = sitk.ImageRegistrationMethod()

    #set similarity metric
    if simMetric == "MeanSquares":
        registration_method.SetMetricAsMeanSquares()
    else:
        print("default sim metric: MeanSquares")
        registration_method.SetMetricAsMeanSquares()

    #set optimizer
    if optimizer == "GradientDescent":
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        registration_method.SetOptimizerScalesFromPhysicalShift()
    else:
        print("default optimizer: GradientDescent")
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        registration_method.SetOptimizerScalesFromPhysicalShift()

    #initial transform
    initial_transform = sitk.TranslationTransform(image1.GetDimension())
    registration_method.SetInitialTransform(initial_transform)

    #set interpolator
    if interpolator == "Linear":
        registration_method.SetInterpolator(sitk.sitkLinear)
    else:
        registration_method.SetInterpolator(sitk.sitkLinear)

    #execute registration
    final_transform = registration_method.Execute(sitk.Cast(image1, sitk.sitkFloat32), sitk.Cast(image2, sitk.sitkFloat32))

    #apply transformation
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image1)
    if samplerInterpolator == "Linear":
        resampler.SetInterpolator(sitk.sitkLinear)
    elif samplerInterpolator == "HigherOrder":
        resampler.SetInterpolator(sitk.sitkBSpline)
    elif samplerInterpolator == "NearestNeighbor":
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        print("default samplerInterpolator Linear")
        resampler.SetInterpolator(sitk.sitkLinear)
    
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(final_transform)

    registered_image = resampler.Execute(image2)
    return registered_image


# Path to the directory that contains the DICOM files
directory1 = "scan1"
directory2 = "scan2"
# Create 3d image with SITK
image1 = data.get_3d_image(directory1)
image2 = data.get_3d_image(directory2)
#does it need to by cast to float32?

registered_image = basic_segment(image1, image2)

data.view_sitk_3d_image(image1, 5, "image1")
data.view_sitk_3d_image(image2, 5, "image2")
data.view_sitk_3d_image(registered_image, 5, "regImage")
