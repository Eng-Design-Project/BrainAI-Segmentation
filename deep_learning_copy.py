import SimpleITK as sitk
import os
import tensorflow as tf
import numpy as np
from scipy.ndimage import convolve
import data  # Your data module

def unet(input_size=(128, 128, 128, 1)):
    inputs = tf.keras.layers.Input(input_size)
    
    conv1 = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    
    conv2 = tf.keras.layers.Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = tf.keras.layers.Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    
    conv3 = tf.keras.layers.Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = tf.keras.layers.Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = tf.keras.layers.Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = tf.keras.layers.Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = tf.keras.layers.Dropout(0.5)(conv4)
    pool4 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    conv5 = tf.keras.layers.Conv3D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = tf.keras.layers.Conv3D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = tf.keras.layers.Dropout(0.5)(conv5)

    up6 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(drop5)
    crop4 = tf.keras.layers.Cropping3D(cropping=((1, 1), (0, 0), (0, 0)))(drop4)
    merge6 = tf.keras.layers.concatenate([crop4, up6], axis=-1)
    conv6 = tf.keras.layers.Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)

    up7 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(conv6)
    merge7 = tf.keras.layers.concatenate([conv3, up7], axis=-1)
    conv7 = tf.keras.layers.Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)

    up8 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(conv7)
    merge8 = tf.keras.layers.concatenate([conv2, up8], axis=-1)
    conv8 = tf.keras.layers.Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)

    up9 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(conv8)
    merge9 = tf.keras.layers.concatenate([conv1, up9], axis=-1)
    conv9 = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)

    outputs = tf.keras.layers.Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)
    
    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def normalizeTF(volume3dDict):
    normalizedDict = {}
    for key, value in volume3dDict.items():
        tensor = tf.convert_to_tensor(value, dtype=tf.float32)
        minVal = tf.reduce_min(tensor)
        maxVal = tf.reduce_max(tensor)
        normalizedTensor = (tensor - minVal) / (maxVal - minVal)
        normalizedDict[key] = normalizedTensor.numpy()
    return normalizedDict

def find_boundary(segment):
    kernel = np.ones((3, 3, 3))
    kernel[1, 1, 1] = 0
    boundary = convolve(segment > 0, kernel) > 0
    boundary = boundary & (segment > 0)
    return boundary

# Your existing dlAlgorithm function
def dlAlgorithm(segmentDict):
    numpyImagesDict = {key: sitk.GetArrayFromImage(img) for key, img in segmentDict.items()}

    # Debugging Step 1: Verify Data Shape
    for key, value in numpyImagesDict.items():
        print(f"Shape of original image array for {key}: {value.shape}")

    normalizedDict = normalizeTF(numpyImagesDict)
    sampleShape = numpyImagesDict[list(numpyImagesDict.keys())[0]].shape  # Assuming all shapes are the same
    
    model = unet(input_size=(*sampleShape, 1))

    # Debugging Step 2: Model Architecture
    model.summary()

    for key, img_array in normalizedDict.items():
        boundary_array = find_boundary(img_array)

        # Debugging Step 3: Verify shapes before training
        print(f"Shape of img_array for training {key}: {img_array.shape}")
        print(f"Shape of boundary_array for training {key}: {boundary_array.shape}")

        img_array = np.expand_dims(np.expand_dims(img_array, axis=0), axis=-1)
        boundary_array = np.expand_dims(np.expand_dims(boundary_array, axis=0), axis=-1)

        # Debugging Step 4: Confirm shape compatibility
        assert img_array.shape == boundary_array.shape, f"Shape mismatch for {key}"

        model.train_on_batch(img_array, boundary_array)

    model.save("unet_model.h5")

# Your existing main code block
if __name__ == "__main__":
    sitk_images_dict = {
        "image1": data.get_3d_image("scan1"),
        "image2": data.get_3d_image("scan2"),
    }
    dlAlgorithm(sitk_images_dict)
