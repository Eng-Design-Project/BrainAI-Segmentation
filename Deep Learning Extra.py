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
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
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


def standardizeTF(volume3dDict):
    standardizedDict = {}
    for key, value in volume3dDict.items():
        tensor = tf.convert_to_tensor(value, dtype=tf.float32)
        mean, stddev = tf.nn.moments(tensor, axes=[0, 1, 2])  # Compute mean and stddev
        standardizedTensor = (tensor - mean) / tf.sqrt(stddev)
        standardizedDict[key] = standardizedTensor.numpy()
    return standardizedDict


def find_boundary(segment):
    kernel = np.ones((3, 3, 3))
    kernel[1, 1, 1] = 0
    boundary = convolve(segment > 0, kernel) > 0
    boundary = boundary & (segment > 0)
    return boundary

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
    normalizedDict = standardizeTF(numpyImagesDict)
    model = unet(input_size=(depth, 128, 128, 1))

    # Callbacks
    early_stopping = EarlyStopping(patience=5, verbose=1)
    model_checkpoint = ModelCheckpoint("best_model.keras", save_best_only=True, verbose=1)
    callbacks_list = [early_stopping, model_checkpoint]
    
    loss_list = []

    for epoch in range(epochs):
        for key, sub_array in normalizedDict.items():
            sub_arrays_split = split_into_subarrays(sub_array, depth)
            
            slice_triplets_to_display = []

            for sub_arr in sub_arrays_split:
                sub_boundary_array = find_boundary(sub_arr)
                sub_arr_exp = np.expand_dims(np.expand_dims(sub_arr, axis=0), axis=-1)
                sub_boundary_array_exp = np.expand_dims(np.expand_dims(sub_boundary_array, axis=0), axis=-1)

                history = model.train_on_batch(sub_arr_exp, sub_boundary_array_exp)
                loss_list.append(history[0])

                # Predict and collect slices for visualization
                pred = model.predict(sub_arr_exp)
                middle_index = depth // 2
                slices_triplets = [
                    (sub_arr[middle_index], pred[0][middle_index])
                ]
                slice_triplets_to_display.extend(slices_triplets)
                
                # Display in groups of three
                if len(slice_triplets_to_display) == 3:
                    show_slices(slice_triplets_to_display)
                    slice_triplets_to_display = []
                    proceed = input("Would you like to see more images? (y/n): ")
                    if proceed.lower() != 'y':
                        return  # Exit the function entirely

            # Display any remaining images
            if slice_triplets_to_display:
                show_slices(slice_triplets_to_display)
                proceed = input("Would you like to proceed to the next epoch? (y/n): ")
                if proceed.lower() != 'y':
                    return  # Exit the function entirely

    # Optional: Load the best saved model
    model = load_model("best_model.keras")
    
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