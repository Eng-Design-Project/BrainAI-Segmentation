import SimpleITK as sitk
import tensorflow as tf
import numpy as np
from scipy.ndimage import convolve
import data
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


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



def get_surrounding_slices(original_slice, sub_arrays, slice_index, depth):
    surrounding_depth = depth // 2
    surrounding_slices = []
    for sub_array in sub_arrays:
        for idx, slice_ in enumerate(sub_array):
            if np.array_equal(slice_, original_slice):
                start_idx = max(0, idx - surrounding_depth)
                end_idx = min(len(sub_array), idx + surrounding_depth + 1)
                surrounding_slices = sub_array[start_idx:end_idx]
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

def dlAlgorithm(segmentDict, depth=5, epochs=3):
    numpyImagesDict = {key: sitk.GetArrayFromImage(img) for key, img in segmentDict.items()}
    normalizedDict = normalizeTF(numpyImagesDict)
    
    # Define the U-Net model
    model = unet(input_size=(depth, 128, 128, 1))

    early_stopping = EarlyStopping(patience=5, verbose=1)
    model_checkpoint = ModelCheckpoint("best_model.keras", save_best_only=True, verbose=1)
    callbacks_list = [early_stopping, model_checkpoint]

    loss_list = []
    all_triplets = []  # <-- Store all the segmented triplets here

    # First loop for segmentation
    for key, sub_array in normalizedDict.items():
        sub_arrays_split = split_into_subarrays(sub_array, depth)
        
        for idx, sub_arr in enumerate(sub_arrays_split):
            surrounding_slices = get_surrounding_slices(sub_arr[depth//2], sub_arrays_split, idx, depth)
            model = load_model("current_model.keras", custom_objects={"weighted_binary_crossentropy": weighted_binary_crossentropy})
            
            sub_boundary_array = find_boundary(surrounding_slices)
            sub_arr_exp = np.expand_dims(np.expand_dims(surrounding_slices, axis=0), axis=-1)
            sub_boundary_array_exp = np.expand_dims(np.expand_dims(sub_boundary_array, axis=0), axis=-1)

            history = model.train_on_batch(sub_arr_exp, sub_boundary_array_exp)
            loss_list.append(history[0])

            pred = model.predict(sub_arr_exp)
            middle_index = depth // 2
            slices_triplet = (surrounding_slices[middle_index], pred[0][middle_index])
            all_triplets.append(slices_triplet)

    # Second loop for displaying the segmented images in batches of 3
    for i in range(0, len(all_triplets), 3):
        batch_triplets = all_triplets[i:i+3]

        for idx, triplet in enumerate(batch_triplets):
            print(f"Triplet {idx + 1} Original Min: {triplet[0].min()}, Max: {triplet[0].max()}")
            print(f"Triplet {idx + 1} Prediction Min: {triplet[1].min()}, Max: {triplet[1].max()}")

        show_slices(batch_triplets)

        proceed = input("Would you like to see more slices? (y/n): ")
        if proceed.lower() != 'y':
            break  # Exit the function entirely

    model = load_model("best_model.keras", custom_objects={"weighted_binary_crossentropy": weighted_binary_crossentropy})
    
    plt.plot(loss_list)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.show()




if __name__ == "__main__":
    sitk_images_dict = {
        "image1": data.get_3d_image("scan1"),
        "image2": data.get_3d_image("scan2"),
    }

    dlAlgorithm(sitk_images_dict)
