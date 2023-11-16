import tensorflow as tf
import numpy as np
from scipy.ndimage import convolve
import data
import matplotlib.pyplot as plt
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from keras.models import load_model
from tensorflow.keras.utils import to_categorical
from PIL import Image




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
def unet(input_size=(5, 128, 128, 1)):  # Notice the change in the last dimension
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

def unet_internal(input_size=(128, 128, 3), num_classes=5): # num_classes should be 4
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


def preprocess_color_atlas(atlas_dir, color_mapping):
    # Initialize an empty list to hold the atlas images
    atlas_images = []

    # Assuming atlas images are stored as individual files in atlas_dir
    for filename in os.listdir(atlas_dir):
        if filename.endswith('.png'):  # or '.jpg', or whatever format they're in
            filepath = os.path.join(atlas_dir, filename)
            image = Image.open(filepath)
            atlas_images.append(np.array(image))

    # Assuming all images have the same shape, stack them into a single numpy array
    atlas_images = np.stack(atlas_images, axis=0)

    labels = np.zeros(atlas_images.shape[:-1], dtype=np.int32)
    
    # Assign labels based on color mapping
    for color, label in color_mapping.items():
        matches = np.all(atlas_images == np.array(color, dtype=np.uint8), axis=-1)
        labels[matches] = label

    # One-hot encode the labels
    labels_one_hot = to_categorical(labels, num_classes=len(color_mapping) + 1)  # +1 for the background
    
    return atlas_images, labels_one_hot


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
    segment_copy = segment.copy()  # Work on a copy to avoid modifying the original
    kernel = np.ones((3, 3, 3))
    kernel[1, 1, 1] = 0
    boundary = convolve(segment_copy > 0, kernel) > 0
    boundary = boundary & (segment_copy > 0)
    segment_copy[boundary] = 1  # Label the boundary as 1
    segment_copy[~boundary] = 0  # Label the rest as 0
    return segment_copy

def show_slices(triplets):
    n = len(triplets)
    fig, axes = plt.subplots(1, n * 2, figsize=(12, 6))
    for i in range(n):
        orig, pred = triplets[i]
        axes[i * 2].imshow(orig.T, cmap="gray", origin="lower")
        # Ensure that pred is 2D when passed to imshow
        pred_2d = np.squeeze(pred)
        if pred_2d.ndim > 2:
            pred_2d = np.max(pred_2d, axis=0)  # Take the max projection along the first axis
        axes[i * 2 + 1].imshow(pred_2d.T, cmap="gray", origin="lower")
    plt.suptitle("Original and Segmented")
    plt.show()

def prepare_data_for_training(segmentDict, depth=5):
    X_train = []
    Y_train = []

    for key, img_array in segmentDict.items():
        sub_arrays = split_into_subarrays(img_array, depth)
        for sub_arr in sub_arrays:
            # Assuming the middle slice of sub_arr is what you want to predict
            middle_slice = sub_arr[depth // 2]

            # Find the boundary of the middle slice
            boundary = find_boundary(middle_slice)

            # Surrounding slices will be your input, and boundary will be your label
            X_train.append(sub_arr)
            Y_train.append(boundary)

    # Convert lists to numpy arrays
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    # Expand dimensions to fit the model's expected input
    X_train = np.expand_dims(X_train, axis=-1)
    Y_train = np.expand_dims(Y_train, axis=-1)

    return X_train, Y_train

def visualize_segmentation(slice, prediction, color_mapping, title="Segmentation"):
    # Convert the prediction to a label map (assuming the prediction is a softmax output)
    label_map = np.argmax(prediction, axis=-1)

    # Verify dimensions match
    assert label_map.shape == slice.shape[:2], "Label map shape does not match slice shape"

    # Convert the label map to a color image
    color_image = np.zeros((slice.shape[0], slice.shape[1], 3), dtype=np.uint8)
    for class_idx, color in color_mapping.items():
        mask = label_map == class_idx
        color_image[mask] = color

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(slice.T, cmap='gray', origin='lower')
    plt.title('Original')

    plt.subplot(1, 2, 2)
    plt.imshow(color_image, origin='lower')  # Make sure to use the correct orientation
    plt.title(title)
    plt.show()




def convert_to_binary_mask(prediction, threshold=0.5):
    binary_mask = (prediction > threshold).astype(np.uint8)
    return binary_mask

# Visualization function for internal segmentation
def visualize_internal_segmentation(all_triplets, color_mapping):
    for original, prediction in all_triplets:
        unique_values = np.unique(prediction)
        print(f"Unique prediction values: {unique_values}")  # Debug print
        if len(unique_values) == 1 and unique_values[0] == 0:
            print("Warning: Predictions are all zero.")

        # Convert the prediction to a label map
        label_map = np.argmax(prediction, axis=-1)

        # Convert the label map to a color image
        color_image = np.zeros((original.shape[0], original.shape[1], 3), dtype=np.uint8)
        for class_idx, color in color_mapping.items():
            color_image[label_map == class_idx] = color

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(original.squeeze(), cmap='gray')
        plt.title('Original')

        plt.subplot(1, 2, 2)
        plt.imshow(color_image)
        plt.title('Segmentation')
        plt.show()




def dlAlgorithm(segmentDict, file_names, depth=5, binary_model_path='brain_skull.keras', 
                multiclass_model_path='internal_segmentation.keras', 
                segmentation_type='internal', 
                X_train=None, Y_train=None, 
                X_train_binary=None, Y_train_binary=None, 
                color_mapping=None, atlas_dir=None,):

    normalizedDict = normalizeTF(segmentDict)
    all_triplets = []  # Define all_triplets here to be used for both binary and internal segmentation

    if segmentation_type == 'skull':
        # Binary Segmentation
        if os.path.exists(binary_model_path):
            print(f"Loading pre-trained binary model from {binary_model_path}...")
            model_binary = load_model(binary_model_path, custom_objects={"weighted_binary_crossentropy": weighted_binary_crossentropy})
        else:
            print("Creating and training a new binary segmentation model...")
            if atlas_dir is not None and color_mapping is not None:
                atlas_images, labels_one_hot = preprocess_color_atlas(atlas_dir, color_mapping)
                model_internal = unet_internal(input_size=(128, 128, 3), num_classes=len(color_mapping))  # Ensure input_size matches here
                history_internal = model_internal.fit(atlas_images, labels_one_hot, validation_split=0.1, epochs=25, batch_size=16)
                model_internal.save(multiclass_model_path)
            else:
                print("Training data for binary segmentation is not provided.")
                return  # Exit the function if training data is not available

        # Binary Segmentation Visualization
        for key, sub_array in zip(file_names, normalizedDict.values()):
            print(f"Processing file: {key}")
            sub_arrays_split = split_into_subarrays(sub_array, depth)
            for idx, sub_arr in enumerate(sub_arrays_split):
                surrounding_slices = get_surrounding_slices(sub_arr[depth//2], sub_arrays_split, depth)
                sub_arr_exp = np.expand_dims(np.expand_dims(surrounding_slices, axis=0), axis=-1)
                pred = model_binary.predict(sub_arr_exp)
                middle_index = depth // 2
                # Convert predictions to binary mask
                binary_mask = convert_to_binary_mask(pred[0][middle_index])
                visualize_segmentation(surrounding_slices[middle_index], binary_mask, title="Binary Segmentation")


    if segmentation_type == 'internal':
        if os.path.exists(multiclass_model_path):
            print(f"Loading pre-trained internal model from {multiclass_model_path}...")
            model_internal = load_model(multiclass_model_path)
        else:
            print("Creating and training a new internal segmentation model...")
            if atlas_dir is not None and color_mapping is not None:
                atlas_images, labels_one_hot = preprocess_color_atlas(atlas_dir, color_mapping)
                model_internal = unet_internal(input_size=(128, 128, 3), num_classes=len(color_mapping) + 1)  # Ensure num_classes is correct
                history_internal = model_internal.fit(atlas_images, labels_one_hot, validation_split=0.1, epochs=25, batch_size=16)
                model_internal.save(multiclass_model_path)
            else:
                print("Training data or color mapping for internal segmentation is not provided.")
                return  # Exit the function if training data or color mapping is not available

        # Internal Segmentation Visualization
        for file_name, sub_array in zip(file_names, normalizedDict.values()):
            print(f"Processing file: {file_name}")
            for slice_idx in range(sub_array.shape[0]):
                single_slice = sub_array[slice_idx, :, :]
                single_slice_3_channels = np.repeat(single_slice[:, :, np.newaxis], 3, axis=2)
                single_slice_expanded = np.expand_dims(single_slice_3_channels, axis=0)
                pred = model_internal.predict(single_slice_expanded)
                pred = np.squeeze(pred)  # Remove the batch dimension
                print("Prediction shape:", pred.shape)  # This should now show (128, 128, 5)
                assert pred.shape == (128, 128, 5), "Prediction shape mismatch"  # Ensure this matches the expected shape
                visualize_segmentation(single_slice, pred, color_mapping, title="Internal Segmentation")
    
        # Ask user if they want to see the trained images after training/loading the model
    # show_images_input = input("Do you want to see the trained images? (yes/no): ").strip().lower()
    # show_images = show_images_input == 'yes'
        # Visualization for all triplets after processing both types
    # Visualization for all triplets after processing both types
    for i in range(0, len(all_triplets), 3):
        batch_triplets = all_triplets[i:i+3]
        show_slices(batch_triplets)

    # Optionally, prompt the user to display the images, if desired
    if all_triplets and input("Show processed images? (y/n): ").strip().lower() == 'y':
        triplet_index = 0
        while triplet_index < len(all_triplets):
            batch_triplets = all_triplets[triplet_index:triplet_index + 3]
            show_slices(batch_triplets)
            triplet_index += len(batch_triplets)
            if triplet_index < len(all_triplets) and input("Show next set of images? (y/n): ").strip().lower() != 'y':
                break



if __name__ == "__main__":
    # Define the atlas directory and color mapping for internal segmentation
    atlas_dir = 'C:\\Users\\Justin Rivera\\OneDrive\\Documents\\ED1\\BrainAI-Segmentation\\atl_segmentation_PNGs\\Brain'
    color_mapping = {
        (236, 28, 36): 0,  # Red - White Matter
        (0, 168, 243): 1,  # Blue - Temporal lobe
        (185, 122, 86): 2, # Brown - Occipital lobe
        (184, 61, 186): 3, # Pink - Frontal Lobe
        # Add other colors and regions as required
    }

    # Example dictionary holding your image data
    sitk_images_dict = {
        "image1": data.get_3d_image("scan1"),
        "image2": data.get_3d_image("scan2"),
    }
    file_names = list(sitk_images_dict.keys())

    # User chooses the segmentation type
    segmentation_type = input("Choose segmentation type ('internal' or 'skull'): ").strip().lower()
    
    if segmentation_type == 'internal':
        # Call dlAlgorithm for internal segmentation
        dlAlgorithm(
            segmentDict=sitk_images_dict,
            file_names=file_names,
            binary_model_path='brain_skull.keras',
            multiclass_model_path='internal_segmentation.keras',
            segmentation_type=segmentation_type,
            atlas_dir=atlas_dir,
            color_mapping=color_mapping,
        )
    elif segmentation_type == 'skull':
        # Call dlAlgorithm for binary segmentation
        dlAlgorithm(
            segmentDict=sitk_images_dict,
            file_names=file_names,
            binary_model_path='brain_skull.keras',
            multiclass_model_path='internal_segmentation.keras',
            segmentation_type=segmentation_type,
        )
    else:
        print("Invalid segmentation type. Please choose 'internal' or 'skull'.")

