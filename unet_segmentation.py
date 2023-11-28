import tensorflow as tf
import numpy as np
from scipy.ndimage import convolve
from skimage.transform import resize
import data
import matplotlib.pyplot as plt
import os
from keras.models import load_model
from tensorflow.keras.utils import to_categorical
import glob
import pydicom
import tkinter as tk
from tkinter import simpledialog


def load_dcm_images_from_folder(folder, target_size=(128, 128)):
    images = []
    for dcm_path in glob.glob(folder + '/*.dcm'):  
        dcm = pydicom.dcmread(dcm_path)
        img = dcm.pixel_array
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        img = resize(img, target_size, preserve_range=True)
        img = normalize_image(img)  # Normalize the image
        images.append(img[..., np.newaxis])  # Add channel dimension
    return np.array(images)


def ensure_directory_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to split 3D images into smaller sub-arrays
def split_into_subarrays(img_array, depth=5):
    total_slices = img_array.shape[0]
    sub_arrays = [img_array[i:i+depth, :, :] for i in range(0, total_slices, depth) if i+depth <= total_slices]
    return sub_arrays

def weighted_binary_crossentropy(y_true, y_pred):
    # custom weights for binary cross entropy
    weight_0 = 1.0  # for regions
    weight_1 = 2.0  # for boundaries
    b_ce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    weight_vector = y_true * weight_1 + (1. - y_true) * weight_0
    weighted_b_ce = weight_vector * b_ce
    return tf.keras.backend.mean(weighted_b_ce)


# Function to create a U-Net model for 3D image segmentation
def unet_generate_model(input_size=(5, 128, 128, 1)):  # Notice the change in the last dimension
    inputs = tf.keras.layers.Input(input_size)
    
# Encoder layers (convolutions and pooling)
    conv1 = tf.keras.layers.Conv3D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = tf.keras.layers.Conv3D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(conv1)
    
    conv2 = tf.keras.layers.Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = tf.keras.layers.Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(conv2)
    
    conv3 = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(conv3)
    
    conv4 = tf.keras.layers.Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = tf.keras.layers.Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = tf.keras.layers.Dropout(0.3)(conv4)
    pool4 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(drop4)
    
    conv5 = tf.keras.layers.Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = tf.keras.layers.Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = tf.keras.layers.Dropout(0.3)(conv5)

    up6 = tf.keras.layers.UpSampling3D(size=(1, 2, 2))(drop5)
    merge6 = tf.keras.layers.concatenate([drop4, up6], axis=-1)
    conv6 = tf.keras.layers.Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)

    up7 = tf.keras.layers.UpSampling3D(size=(1, 2, 2))(conv6)
    merge7 = tf.keras.layers.concatenate([conv3, up7], axis=-1)
    conv7 = tf.keras.layers.Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)

    up8 = tf.keras.layers.UpSampling3D(size=(1, 2, 2))(conv7)
    merge8 = tf.keras.layers.concatenate([conv2, up8], axis=-1)
    conv8 = tf.keras.layers.Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)

    up9 = tf.keras.layers.UpSampling3D(size=(1, 2, 2))(conv8)
    merge9 = tf.keras.layers.concatenate([conv1, up9], axis=-1)
    conv9 = tf.keras.layers.Conv3D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    
    # Define output layer
    outputs = tf.keras.layers.Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)
    
    # Compile and return the model
    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=weighted_binary_crossentropy, metrics=['accuracy'])
    return model

def unet_internal(input_size=(128, 128, 1), num_classes=5):
    inputs = tf.keras.layers.Input(input_size)

    # Encoder layers (convolutions and pooling)
    conv1 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = tf.keras.layers.Dropout(0.3)(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = tf.keras.layers.Dropout(0.3)(conv5)

    # Decoder layers (up-sampling and concatenation)
    up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(drop5)
    merge6 = tf.keras.layers.concatenate([drop4, up6], axis=-1)
    conv6 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)

    up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
    merge7 = tf.keras.layers.concatenate([conv3, up7], axis=-1)
    conv7 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)

    up8 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7)
    merge8 = tf.keras.layers.concatenate([conv2, up8], axis=-1)
    conv8 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)

    up9 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8)
    merge9 = tf.keras.layers.concatenate([conv1, up9], axis=-1)
    conv9 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)

    # Output layer
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='softmax')(conv9)

    # Compile the model
    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_surrounding_slices(original_slice, sub_arrays, depth):
    surrounding_depth = depth // 2
    surrounding_slices = []
    for sub_array in sub_arrays:
        for idx, slice_ in enumerate(sub_array):
            if np.array_equal(slice_, original_slice):
                start_idx = max(0, idx - surrounding_depth)
                end_idx = min(len(sub_array), idx + surrounding_depth + 1)
                surrounding_slices = sub_array[start_idx:end_idx]
                break  # You can break here since you've found the slice and don't need to check further
    if len(surrounding_slices) != depth:
        # Handle edge cases by padding with zeros
        padding_slices = depth - len(surrounding_slices)
        pad_before = padding_slices // 2
        pad_after = padding_slices - pad_before
        surrounding_slices = np.pad(surrounding_slices, ((pad_before, pad_after), (0, 0), (0, 0)), 'constant')
    return surrounding_slices


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
    # Check if the segment is 2D or 3D and choose the kernel accordingly
    if segment.ndim == 3:  # 3D data
        kernel = np.ones((3, 3, 3))
    elif segment.ndim == 2:  # 2D data
        kernel = np.ones((3, 3))
    else:
        raise ValueError("Unsupported segment dimensions")

    kernel[(kernel.shape[0] // 2, kernel.shape[1] // 2)] = 0
    segment_copy = segment.copy()
    boundary = convolve(segment_copy > 0, kernel) > 0
    boundary = boundary & (segment_copy > 0)
    segment_copy[boundary] = 1  # Label the boundary as 1
    segment_copy[~boundary] = 0  # Label the rest as 0
    return segment_copy

'''def show_slices(triplets): This returns the images in color. This was an accident but they looked cool so I just commented it out insteaad of deleting it
    n = len(triplets)
    fig, axes = plt.subplots(1, n, figsize=(n * 6, 6))
    for i in range(n):
        orig, pred = triplets[i]
        axes[i].imshow(orig.T, cmap="gray", origin="lower")  # Display the original image
         # Ensure the prediction is 2D
        if pred.ndim > 2:
             pred = np.squeeze(pred)  # Remove any singleton dimensions
         # In case pred is still not 2D, take its max projection across the last dimension
        if pred.ndim > 2:
             pred = np.max(pred, axis=-1)
        axes[i].imshow(pred.T, cmap="jet", alpha=0.5, origin="lower")  # Overlay the prediction
    plt.suptitle("Original and Segmented")
    plt.show()
'''

def show_slices(triplets, threshold=0.5, brightening_factor=1.3):  # Adjust threshold and brightening factor as needed
    n = len(triplets)
    fig, axes = plt.subplots(2, n, figsize=(2*n, 4))

    for i, (orig, pred) in enumerate(triplets):
        orig = np.squeeze(orig).astype(np.float32)  # Ensure original is float32 to prevent clipping
        if orig.ndim != 2:
            raise ValueError(f"Original image has unexpected dimensions: {orig.shape}")

        # Prediction processing
        if pred.ndim == 3 and pred.shape[-1] > 1:
            pred = np.argmax(pred, axis=-1)
        elif pred.ndim == 3 and pred.shape[-1] == 1:
            pred = np.squeeze(pred)
        elif pred.ndim != 2:
            raise ValueError(f"Prediction has unexpected dimensions: {pred.shape}")

        binary_mask = pred > threshold

        # Apply brightening
        brightened_image = np.copy(orig)
        brightened_image[binary_mask] *= brightening_factor
        brightened_image = np.clip(brightened_image, 0, np.max(orig))  # Clip to the max of original to prevent overexposure

        axes[0, i].imshow(orig, cmap="gray")
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')

        axes[1, i].imshow(brightened_image, cmap="gray")  # Show the brightened image
        axes[1, i].set_title("Segmented")
        axes[1, i].axis('off')

    plt.show()



def get_unet_result_coordinates(original, new):

    coordinates_dict = {}
    #We want all coordinates NOT 
    for key in original.keys():
        coordinate_list = []
        for x in range(original[key].size[0]):
            for y in range(original[key].size[1]):
                for z in range(original[key].size[2]):
                    if not new[key][x,y,z] > original[key][x,y,z]:
                        coordinate_list.append([x,y,z])
        coordinates_dict[key] = coordinate_list

    return coordinates_dict


def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val > 0:
        normalized_image = (image - min_val) / (max_val - min_val)
    else:
        normalized_image = image - min_val
    return normalized_image


def prepare_data_for_training(subarrays, depth=5, num_classes=5):
    X_train = []
    Y_train = []

    for sub_arr in subarrays:
        middle_slice = sub_arr[depth // 2]

        # Generate the boundary and one-hot encode it
        boundary = find_boundary(middle_slice)
        boundary_one_hot = to_categorical(boundary, num_classes=num_classes)

        X_train.append(middle_slice[..., np.newaxis])  # Add channel dimension
        Y_train.append(boundary_one_hot)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train).reshape(-1, 128, 128, num_classes)

    return X_train, Y_train


def get_user_selection(region_options):
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    region_selection = simpledialog.askinteger("Select Region",
                                               "Select the region to visualize:\n" +
                                               "\n".join([f"{k}: {v}" for k, v in region_options.items()]),
                                               parent=root)
    return region_selection

def dlAlgorithm(segmentDict, file_names, depth=5, binary_model_path='my_model.keras',
                internal_folder_paths=None, segmentation_type='internal'):

    normalizedDict = normalizeTF(segmentDict) if segmentDict is not None else None
    all_triplets = []

    if segmentation_type == 'skull':
        if os.path.exists(binary_model_path):
            model_binary = load_model(binary_model_path, custom_objects={"weighted_binary_crossentropy": weighted_binary_crossentropy})
            for key, sub_array in zip(file_names, normalizedDict.values()):
                sub_arrays_split = split_into_subarrays(sub_array, depth)
                for idx, sub_arr in enumerate(sub_arrays_split):
                    surrounding_slices = get_surrounding_slices(sub_arr[2], sub_arrays_split, depth)
                    sub_arr_exp = np.expand_dims(np.expand_dims(surrounding_slices, axis=0), axis=-1)
                    pred = model_binary.predict(sub_arr_exp)
                    middle_index = depth // 2
                    slices_triplet = (surrounding_slices[middle_index], pred[0][middle_index])
                    all_triplets.append(slices_triplet)
        else:
            print("Binary segmentation model path does not exist. Please check the path and try again.")
            return

    elif segmentation_type == 'internal':
        
        for region_name, folder_path in internal_folder_paths.items():
            print(f"Processing region: {region_name}")

            model_path = os.path.join(folder_path, f"{region_name.lower().replace(' ', '_')}_model.keras")
            if not os.path.exists(model_path):
                print(f"Training new model for {region_name}...")
                images = load_dcm_images_from_folder(folder_path)
                X_train, Y_train = prepare_data_for_training(images, depth=depth, num_classes=5)
                model = unet_internal(input_size=(128, 128, 1), num_classes=5)
                model.fit(X_train, Y_train, epochs=25, batch_size=16)
                ensure_directory_exists(folder_path)
                model.save(model_path)
            else:
                print(f"Loading model for {region_name} from {model_path}...")
                model = load_model(model_path)
            
            images = load_dcm_images_from_folder(folder_path)
            for slice_idx in range(images.shape[0]):  # Iterate through each slice in the loaded images
                slice_2d = images[slice_idx, :, :, 0]  
                slice_2d_normalized = normalize_image(slice_2d)
                slice_2d_normalized = np.expand_dims(slice_2d_normalized, axis=-1)
                slice_2d_normalized = np.expand_dims(slice_2d_normalized, axis=0)
                pred = model.predict(slice_2d_normalized)
                all_triplets.append((slice_2d, pred[0])) 

            # Display images for the current region
            #display_images_for_region(all_triplets, region_name)
            all_triplets.clear()  # Clear the list for the next region

    print("All images have been processed.")



def execute_unet(inputDict, depth=5):

    dict_of_3d_arrays = {}

    if isinstance(inputDict, dict):
            print("input is a dictionary.")
            dict_of_3d_arrays = inputDict
    else:
        print("input is an array.")
        dict_of_3d_arrays["FullScan"] = inputDict

    normalizedDict = normalizeTF(dict_of_3d_arrays)
    all_triplets = []
    model_paths = {key: f"{key}_model.keras" for key in normalizedDict.keys()}

    for key, array3d in normalizedDict.items():
        if os.path.exists(model_paths[key]):
            # If the path exists
            print(f"The path for '{key}' exists.")
            subarrays_split = split_into_subarrays(array3d)
            model_binary = load_model(model_paths[key], custom_objects={"weighted_binary_crossentropy": weighted_binary_crossentropy})
            for idx, sub_arr in enumerate(subarrays_split):
                surrounding_slices = get_surrounding_slices(sub_arr[2], subarrays_split, depth)
                sub_arr_exp = np.expand_dims(np.expand_dims(surrounding_slices, axis=0), axis=-1)
                pred = model_binary.predict(sub_arr_exp)
                middle_index = depth // 2
                slices_triplet = (surrounding_slices[middle_index], pred[0][middle_index])
                all_triplets.append(slices_triplet)
        else:
            # If the path does not exist
            print(f"The path for '{key}' does not exist.")
            model = unet_generate_model()
            subarrays = split_into_subarrays(array3d)
            print(f"Training new model for {key}...")
            X_train, Y_train = prepare_data_for_training(subarrays)
            model.fit(X_train, Y_train, epochs=25, batch_size=16)
            model.save(model_paths[key])


    
def display_images_for_region(all_triplets, region_name):
    print(f"Displaying images for {region_name}...")
    triplet_index = 0
    while triplet_index < len(all_triplets):
        batch_triplets = all_triplets[triplet_index:triplet_index + 3]
        show_slices(batch_triplets)  
        triplet_index += 3

        if triplet_index < len(all_triplets):
            print(f"Continuing with more slices from {region_name}...")




if __name__ == "__main__":

    # Example dictionary holding your image data for skull segmentation
    sitk_images_dict = {
        "image1": data.get_3d_image("scan1"),
        "image2": data.get_3d_image("scan2"),
    }
    execute_unet(sitk_images_dict)




