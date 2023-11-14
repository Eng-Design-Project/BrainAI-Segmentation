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

def unet_internal(input_size=(5, 128, 128, 1), num_classes=4):
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

    # The output layer should have 'num_classes' channels and a 'softmax' activation
    outputs = tf.keras.layers.Conv3D(num_classes, (1, 1, 1), activation='softmax')(conv9)
    
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
        pred_2d = np.squeeze(pred.T)
        axes[i * 2 + 1].imshow(pred_2d, cmap="gray", origin="lower")
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


# Change file path to your own file path to my_model.keras
def dlAlgorithm(segmentDict, depth=5, binary_model_path='brain_skull.keras', multiclass_model_path='internal_segmentation.keras', segmentation_type='internal', atlas_dir='', color_mapping=None):
    normalizedDict = normalizeTF(segmentDict)
    
    if segmentation_type == 'internal':
        # Load or create the internal segmentation model
        if os.path.exists(multiclass_model_path):
            print("Loading pre-trained multiclass model...")
            model = load_model(multiclass_model_path)
        else:
            print("Creating and training a new internal segmentation model...")
            model = unet_internal(input_size=(depth, 128, 128, 1), num_classes=len(color_mapping) + 1)
            
            atlas_images, labels_one_hot = preprocess_color_atlas(atlas_dir, color_mapping)
            
            # Here, you should adapt the data preparation to match the multiclass segmentation problem
            # For example:
            # X_train = atlas_images
            # Y_train = labels_one_hot
            
            # Train the model
            history = model.fit(X_train, Y_train, validation_split=0.1, epochs=50, batch_size=16)
            
            # Save the trained model
            model.save(multiclass_model_path)
            
            # Optionally, plot the training loss and validation loss
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='validation')
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            plt.show()

    # Prediction and visualization block, executed regardless of whether the model was trained or loaded
    all_triplets = []
    for key, sub_array in normalizedDict.items():
        sub_arrays_split = split_into_subarrays(sub_array, depth)
        for idx, sub_arr in enumerate(sub_arrays_split):
            surrounding_slices = get_surrounding_slices(sub_arr[depth//2], sub_arrays_split, depth)
            sub_arr_exp = np.expand_dims(np.expand_dims(surrounding_slices, axis=0), axis=-1)
            
            pred = model.predict(sub_arr_exp)
            middle_index = depth // 2
            slices_triplet = (surrounding_slices[middle_index], pred[0][middle_index])
            all_triplets.append(slices_triplet)
    
    # Display the segmented images
    triplet_index = 0
    while True:
        batch_triplets = all_triplets[triplet_index:triplet_index + 3]
        for idx, triplet in enumerate(batch_triplets):
            print(f"Triplet {idx + 1 + triplet_index} Original Min: {triplet[0].min()}, Max: {triplet[0].max()}")
            print(f"Triplet {idx + 1 + triplet_index} Segmented Min: {triplet[1].min()}, Max: {triplet[1].max()}")
        show_slices(batch_triplets)

        triplet_index += 3

        if triplet_index >= len(all_triplets):
            print("End of list.")
            restart = input("Would you like to start from the beginning? (y/n): ")
            if restart.lower() == 'y':
                triplet_index = 0  # Reset index to start from the beginning
                continue  # Continue the loop from the beginning
            else:
                break  # Exit the loop if the user does not want to continue
        else:
            proceed = input("Would you like to see more slices? (y/n): ")
            if proceed.lower() != 'y':
                break  # Exit the loop if the user does not want to continue

# Visualization function for internal segmentation
def visualize_internal_segmentation(all_triplets, color_mapping):
    for original, prediction in all_triplets:
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(original, cmap='gray')
        plt.title('Original')
        
        plt.subplot(1, 2, 2)
        # Convert predictions to color images using color mapping
        prediction_colored = np.zeros((*prediction.shape, 3), dtype=np.uint8)
        for class_idx, color in enumerate(color_mapping):
            prediction_colored[prediction == class_idx] = color
        plt.imshow(prediction_colored)
        plt.title('Segmentation')
        plt.show()

if __name__ == "__main__":
    # Define the atlas directory and color mapping
    atlas_dir = 'C:\\Users\\Justin Rivera\\OneDrive\\Documents\\ED1\\BrainAI-Segmentation\\Color Atlas internal'
    segmentation_type = 'internal'  # Or 'skull' depending on what you want to run
    
    # Define the color to region mapping here
    color_mapping = {
        (237, 28, 36): 0,  # Red - Temporal lobe
        (0, 162, 232): 1,  # Blue - Frontal lobe
        (0, 255, 0): 2,    # Green - Occipital lobe
        # Add other colors and regions as required
    }

    # Example dictionary holding your image data
    sitk_images_dict = {
        "image1": data.get_3d_image("scan1"),
        "image2": data.get_3d_image("scan2"),
    }

    # Load and preprocess atlas images if internal segmentation is needed
    if segmentation_type == 'internal':
        atlas_images, labels_one_hot = preprocess_color_atlas(atlas_dir, color_mapping)

    # Call dlAlgorithm with the appropriate parameters
    dlAlgorithm(
        sitk_images_dict,
        binary_model_path='brain_skull.keras',
        multiclass_model_path='internal_segmentation.keras',
        segmentation_type=segmentation_type,
        atlas_dir=atlas_dir,  # This is now redundant and can be removed if not used in dlAlgorithm
        color_mapping=color_mapping
    )

