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
import glob
from skimage.io import imread
from skimage.transform import resize

def load_images_from_folder(folder, target_size=(128, 128)):
    images = []
    for img_path in glob.glob(folder + '/*.png'):  # assuming images are in PNG format
        img = imread(img_path, as_gray=True)  # Load as grayscale
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

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    # Avoid division by zero if the image is constant
    if max_val - min_val > 0:
        normalized_image = (image - min_val) / (max_val - min_val)
    else:
        normalized_image = image - min_val  # Will result in an image of zeros
    return normalized_image


def prepare_data_for_training(img_array, depth=5, num_classes=5):
    X_train = []
    Y_train = []

    sub_arrays = [img_array[i:i + depth] for i in range(0, img_array.shape[0], depth) if i + depth <= img_array.shape[0]]
    for sub_arr in sub_arrays:
        middle_slice = sub_arr[depth // 2]

        # Generate the boundary and one-hot encode it
        boundary = find_boundary(middle_slice)
        boundary_one_hot = to_categorical(boundary, num_classes=num_classes)

        X_train.append(middle_slice[..., np.newaxis])  # Add channel dimension
        Y_train.append(boundary_one_hot)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train).reshape(-1, 128, 128, num_classes)

    return X_train, Y_train


def visualize_segmentation(slice, prediction, title="Segmentation"):
    # Squeeze the prediction to remove the batch dimension
    prediction = np.squeeze(prediction)
    
    # If the prediction has more than two dimensions, take the maximum projection across channels
    if prediction.ndim > 2:
        prediction = np.max(prediction, axis=-1)

    # Normalize prediction to the range [0, 255] for visualization
    prediction = (prediction * 255).astype(np.uint8)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(slice.T, cmap='gray', origin='lower')
    plt.title('Original')

    plt.subplot(1, 2, 2)
    plt.imshow(prediction.T, cmap='gray', origin='lower')
    plt.title(title)
    plt.show()



# def convert_to_binary_mask(prediction, threshold=0.5):
#     binary_mask = (prediction > threshold).astype(np.uint8)
#     return binary_mask

# Visualization function for internal segmentation
def visualize_internal_segmentation(original, prediction, title="Segmentation"):
    plt.figure(figsize=(12, 6))
    
    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    
    # Prediction Overlay
    plt.subplot(1, 3, 2)
    plt.imshow(original, cmap='gray')
    plt.imshow(prediction, alpha=0.5)
    plt.title('Prediction Overlay')

    # Prediction Map
    plt.subplot(1, 3, 3)
    plt.imshow(prediction)
    plt.title('Prediction Map')
    
    plt.suptitle(title)
    plt.show()



def dlAlgorithm(segmentDict, file_names, depth=5, binary_model_path='my_model.keras',
                internal_folder_paths=None, segmentation_type='internal', training_data=None):

    normalizedDict = normalizeTF(segmentDict)
    all_triplets = []

    
    if segmentation_type == 'skull':
        if os.path.exists(binary_model_path):
            print(f"Loading pre-trained binary model from {binary_model_path}...")
            model_binary = load_model(binary_model_path, custom_objects={"weighted_binary_crossentropy": weighted_binary_crossentropy})
        else:
            print("Creating and training a new binary segmentation model...")

        for key, sub_array in zip(file_names, normalizedDict.values()):
            print(f"Processing file: {key}")
            sub_arrays_split = split_into_subarrays(sub_array, depth)
            for idx, sub_arr in enumerate(sub_arrays_split):
                surrounding_slices = get_surrounding_slices(sub_arr[depth // 2], sub_arrays_split, depth)
                sub_arr_exp = np.expand_dims(np.expand_dims(surrounding_slices, axis=0), axis=-1)
                pred = model_binary.predict(sub_arr_exp)
                middle_index = depth // 2
                all_triplets.append((surrounding_slices[middle_index], pred[0][middle_index]))

    elif segmentation_type == 'internal':
        models_internal = {}
        region_options = {
            1: "Frontal Lobe",
            2: "Temporal Lobe",
            3: "Occipital Lobe",
            4: "White Matter"
        }

        # Load or train models for each region
        for region, folder_path in internal_folder_paths.items():
            images = load_images_from_folder(folder_path)  # Always load images
            X_train, Y_train = prepare_data_for_training(images, depth=depth, num_classes=5)  # Prepare data
            
            model_path = os.path.join(folder_path, f"{region.lower().replace(' ', '_')}_model.keras")
            if os.path.exists(model_path):
                print(f"Loading model for {region} from {model_path}...")
                model = load_model(model_path)
            else:
                print(f"Training new model for {region}...")
                model = unet_internal(input_size=(128, 128, 1), num_classes=5)
                model.fit(X_train, Y_train, epochs=10, batch_size=16)
                ensure_directory_exists(folder_path)
                model.save(model_path)
            models_internal[region] = model

        # Visualization part
        while True:
            try:
                region_selection = int(input("Select the region to visualize (1-Frontal, 2-Temporal, 3-Occipital, 4-White Matter): ").strip())
                region_to_view = region_options.get(region_selection, None)
            except ValueError:
                print("Invalid input. Please enter a number between 1 and 4.")
                continue

            if region_to_view in models_internal:
                model = models_internal[region_to_view]
                for key, img_3d in segmentDict.items():
                    print(f"Processing file: {key} for {region_to_view}")
                    for slice_idx in range(img_3d.shape[2]):
                        slice_2d = img_3d[:, :, slice_idx]
                        # Resize each slice to (128, 128) if not already
                        if slice_2d.shape != (128, 128):
                            slice_2d = resize(slice_2d, (128, 128), preserve_range=True)
                        slice_2d_normalized = np.expand_dims(np.expand_dims(slice_2d, axis=0), axis=-1) / 255.0
                        pred = model.predict(slice_2d_normalized)
                        visualize_segmentation(slice_2d, pred[0], title=f"Segmentation - {region_to_view}")
            else:
                print("Region not recognized. Please enter a number between 1 and 4.")

            continue_viewing = input("Would you like to visualize another region? (y/n): ").strip().lower()
            if continue_viewing != 'y':
                break


    # Optionally, prompt the user to display the images, if desired
    if all_triplets and input("Show processed images? (y/n): ").strip().lower() == 'y':
        for original, prediction in all_triplets:
            visualize_segmentation(original, prediction, title="Segmentation Results")




if __name__ == "__main__":
    # Paths to the folders containing images for each brain region
    internal_folder_paths = {
        "Frontal Lobe": "C:\\Users\\Justin Rivera\\OneDrive\\Documents\\ED1\\BrainAI-Segmentation\\Internal Segment Images Unet\\Frontal",
        "Temporal Lobe": "C:\\Users\\Justin Rivera\\OneDrive\\Documents\\ED1\\BrainAI-Segmentation\\Internal Segment Images Unet\\Occipital",
        "Occipital Lobe": "C:\\Users\\Justin Rivera\\OneDrive\\Documents\\ED1\\BrainAI-Segmentation\\Internal Segment Images Unet\\Temporal",
        "White Matter": "C:\\Users\\Justin Rivera\\OneDrive\\Documents\\ED1\\BrainAI-Segmentation\\Internal Segment Images Unet\\White Matter"
    }

    # Example dictionary holding your image data for skull segmentation
    sitk_images_dict = {
        "image1": data.get_3d_image("scan1"),
        "image2": data.get_3d_image("scan2"),
    }
    file_names = list(sitk_images_dict.keys())

    segmentation_type = input("Choose segmentation type ('internal' or 'skull'): ").strip().lower()
    
    if segmentation_type == 'internal':
        dlAlgorithm(
            segmentDict=sitk_images_dict,  # Or however you plan to load images for internal segmentation
            file_names=file_names,
            internal_folder_paths=internal_folder_paths,  # Added for internal segmentation
            segmentation_type=segmentation_type
        )
    elif segmentation_type == 'skull':
        dlAlgorithm(
            segmentDict=sitk_images_dict,
            file_names=file_names,
            binary_model_path='my_model.keras',  # Assuming this is your skull segmentation model
            segmentation_type=segmentation_type
        )
    else:
        print("Invalid segmentation type. Please choose 'internal' or 'skull' segmentation")


