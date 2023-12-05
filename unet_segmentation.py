import tensorflow as tf
import numpy as np
from scipy.ndimage import convolve
import data
import matplotlib.pyplot as plt
import os
from keras.models import load_model

# Function to split a 3D image into smaller sub-arrays of a specified depth.
# This is useful for processing large 3D images in smaller chunks.
def split_into_subarrays(img_array, depth=5):
    total_slices = img_array.shape[0]
    sub_arrays = [img_array[i:i+depth, :, :] for i in range(0, total_slices, depth) if i+depth <= total_slices]
    return sub_arrays

# Custom loss function for binary cross-entropy with optional weighting.
# This can be used to give different importance to different classes during training.
def weighted_binary_crossentropy(y_true, y_pred):
    b_ce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    # Modify this function if you need to apply weights to the binary cross-entropy.
    return b_ce


# Function to create a U-Net model for 3D image segmentation
# U-Net is a convolutional network architecture for fast and precise segmentation of both whole brains and segments.
def unet_generate_model(input_size=(5, 128, 128, 1)): 
    inputs = tf.keras.layers.Input(input_size)
    
    # Encoder part: series of convolutions and pooling layers to capture features.
    # Each block contains Convolution -> Activation -> Convolution -> MaxPooling.
    # The number of filters doubles with each block.
    # Decoder part: series of upsampling and concatenation with corresponding encoder outputs.
    # Each block contains UpSampling -> Concatenation -> Convolution -> Activation.
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
    # Final layer: Convolution to get to the desired number of output channels.
    outputs = tf.keras.layers.Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)
    
    # Compile and return the model
    # Creating the model and compiling it with the defined optimizer, loss function, and metrics.
    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=weighted_binary_crossentropy, metrics=['accuracy'])
    return model

# Function to generate predictions for each subarray using the provided model.
# Useful for processing each chunk of the large 3D image separately.
def generate_predictions(subarrays, model):
    predictions = []
    correct_shape_count = sum(1 for sub_arr in subarrays if sub_arr.shape == (5, 128, 128))

    if correct_shape_count == 0:
        print("No subarrays with the correct shape.")
    if not subarrays:
        print("Input subarrays list is empty.")

    # Iterate over each subarray, reshape it to the model's expected input shape,
    # perform the prediction, and store the result.
    for sub_arr in subarrays:
        # Check if the subarray has the correct shape for the model
        if sub_arr.shape == (5, 128, 128):
            # Reshape the subarray to include the batch size and channel dimensions
            sub_arr_reshaped = np.expand_dims(sub_arr, axis=0)  # Adds the batch size dimension
            sub_arr_reshaped = np.expand_dims(sub_arr_reshaped, axis=-1)  # Adds the channel dimension

            # Generate prediction
            pred = model.predict(sub_arr_reshaped)

            # Keep the prediction in 4D (including the channel dimension) and store it
            pred_4d = pred[0, :, :, :, :]  # Shape: [depth, height, width, channels]
            predictions.append(pred_4d)
        else:
            print("Subarray with incorrect shape encountered:", sub_arr.shape)
            continue

    if len(predictions) > 0:
        # Concatenate all predictions along the depth axis
        combined_prediction = np.concatenate(predictions, axis=0)
    
    else:
        print("predictions is empty")
        combined_prediction = np.array([])

    print("generate predictions complete")
    print(combined_prediction.shape)
    return combined_prediction


def get_surrounding_slices(original_slice, sub_arrays, depth):
    surrounding_depth = depth // 2
    surrounding_slices = []
    for sub_array in sub_arrays:
        for idx, slice_ in enumerate(sub_array):
            if np.array_equal(slice_, original_slice):
                start_idx = max(0, idx - surrounding_depth)
                end_idx = min(len(sub_array), idx + surrounding_depth + 1)
                surrounding_slices = sub_array[start_idx:end_idx]
                break  
    if len(surrounding_slices) != depth:
        # Handle edge cases by padding with zeros
        padding_slices = depth - len(surrounding_slices)
        pad_before = padding_slices // 2
        pad_after = padding_slices - pad_before
        surrounding_slices = np.pad(surrounding_slices, ((pad_before, pad_after), (0, 0), (0, 0)), 'constant')
    return surrounding_slices

# Function to normalize 3D volumes.
# Normalization is crucial for consistent model training and prediction.
def normalizeTF(volume3dDict):
    normalizedDict = {}
    # For each 3D volume, convert it to a TensorFlow tensor, find min and max,
    # and normalize the volume to a 0-1 range.
    for key, value in volume3dDict.items():
        tensor = tf.convert_to_tensor(value, dtype=tf.float32)
        minVal = tf.reduce_min(tensor)
        maxVal = tf.reduce_max(tensor)
        normalizedTensor = (tensor - minVal) / (maxVal - minVal)
        normalizedDict[key] = normalizedTensor.numpy()
    return normalizedDict

# Function to find the boundary of a segmented region.
# Used in post-processing to delineate the edges of segmented regions.
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

# Function to generate coordinates below a certain threshold from the 3D prediction.
def get_unet_result_coordinates(predict_3d, threshold=0.4):
    coordinates_list = []
    # Get coordinates below the threshold
    for x in range(predict_3d.shape[0]):
        for y in range(predict_3d.shape[1]):
            for z in range(predict_3d.shape[2]):
                if predict_3d[x, y, z] < threshold:  # Assuming the last dimension is the channel
                    coordinates_list.append([x, y, z])
    return coordinates_list


# Processes the subarrays to create training samples and their corresponding labels.
def prepare_data_for_training(subarrays, depth=5):
    X_train = []
    Y_train = []

    for sub_arr in subarrays:
        if sub_arr.shape[0] == depth:
            # Generate the boundary for the middle slice
            #middle_slice = sub_arr[depth // 2]
            boundary = find_boundary(sub_arr)

            # Add channel dimension to each slice in the sub-array and to the boundary
            sub_arr_processed = sub_arr[..., np.newaxis]
            boundary_processed = boundary[..., np.newaxis]

            X_train.append(sub_arr_processed)
            Y_train.append(boundary_processed)

    # Convert lists to numpy arrays
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    print(X_train.shape)
    print(Y_train.shape)

    return X_train, Y_train

# Main function to execute the U-Net process on the input data.
# Orchestrates the whole process of loading data, training/predicting with U-Net, and processing results.
def execute_unet(inputDict, depth=5):
    dict_of_3d_arrays = {}
    new_dict = {}

    if isinstance(inputDict, dict):
        print("input is a dictionary.")
        dict_of_3d_arrays = inputDict
    else:
        print("input is an array.")
        dict_of_3d_arrays["FullScan"] = inputDict

    normalizedDict = normalizeTF(dict_of_3d_arrays)
    model_paths = {key: f"{key}_model.keras" for key in normalizedDict.keys()}

    final_output = {}

    for key, array3d in normalizedDict.items():
        if os.path.exists(model_paths[key]):
            # Path exists
            print(array3d.shape)
            data.display_3d_array_slices(array3d, 5)
            print(f"The path for '{key}' exists.")
            subarrays_split = split_into_subarrays(array3d)
            model_binary = load_model(model_paths[key], custom_objects={"weighted_binary_crossentropy": weighted_binary_crossentropy})
            predict_3d = generate_predictions(subarrays_split, model_binary)
            
            coordinates_below_threshold = get_unet_result_coordinates(predict_3d)
            final_output[key] = coordinates_below_threshold
        else:
            # Path does not exist
            print(f"The path for '{key}' does not exist.")
            model = unet_generate_model()
            subarrays = split_into_subarrays(array3d)
            print(subarrays[0].shape)
            print(f"Training new model for {key}...")
            X_train, Y_train = prepare_data_for_training(subarrays)
            model.fit(X_train, Y_train, epochs=10, batch_size=16)
            model.save(model_paths[key])
            predict_3d = generate_predictions(subarrays, model)
            coordinates_below_threshold = get_unet_result_coordinates(predict_3d)
            final_output[key] = coordinates_below_threshold

    print("Final output with coordinates below the threshold:", final_output)
    return final_output
            


if __name__ == "__main__":
    # Example dictionary holding your image data for skull segmentation
    sitk_images_dict = {
        "image1": data.get_3d_image("scan1"),
        #"image2": data.get_3d_image("scan2"),
    }
    results = data.set_seg_results_with_dir(r"C:\\Users\\Justin Rivera\\OneDrive\\Documents\\ED1\\Testing Atlas seg Unet.DCMs")
    final_output = execute_unet(data.segmentation_results)
    print(final_output)
    print(results)
