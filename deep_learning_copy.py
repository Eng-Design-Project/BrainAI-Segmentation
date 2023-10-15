import SimpleITK as sitk  # Medical imaging
import tensorflow as tf  # Deep learning
import numpy as np  # Numerical operations
from scipy.ndimage import convolve  # Convolution
import data  # Custom data module
from tensorflow.keras.models import load_model  # Keras model loading
import matplotlib.pyplot as plt  # Visualization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Function to split 3D images into smaller sub-arrays
def split_into_subarrays(img_array, depth=5):

#Splits a 3D image into sub-arrays of given depth.
    total_slices = img_array.shape[0]
    # Create list of sub-arrays
    sub_arrays = [img_array[i:i+depth, :, :] for i in range(0, total_slices, depth) if i+depth <= total_slices]
    return sub_arrays

# Function to create a U-Net model for 3D image segmentation
def unet(input_size=(5, 128, 128, 2)):  # Notice the change in the last dimension
    
    inputs = tf.keras.layers.Input(input_size)  # Define input layer

    
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

def show_slices(slices):
    """ Function to display a batch of image slices """
    n = len(slices)
    fig, axes = plt.subplots(1, n * 2, figsize=(12, 6))
    
    for i in range(n):
        orig, pred = slices[i]
        axes[i * 2].imshow(orig.T, cmap="gray", origin="lower")
        axes[i * 2 + 1].imshow(pred.T, cmap="gray", origin="lower")
        
    plt.suptitle("Middle slices of original and prediction")
    plt.show()


def dlAlgorithm(segmentDict, atlasDict, depth=5, epochs=3):
    numpyImagesDict = {key: sitk.GetArrayFromImage(img) for key, img in segmentDict.items()}
    normalizedDict = normalizeTF(numpyImagesDict)
    atlasDict = normalizeTF(atlasDict)  # Assuming you also normalize atlas images
    
    model = unet(input_size=(depth, 128, 128, 2))  # Now expects 2 channels: atlas and target
    
    loss_list = []
    accumulated_slices = []
    
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")
        
        for scan_name, img_array in normalizedDict.items():
            print(f"Processing {scan_name}...")
            
            target_sub_arrays = split_into_subarrays(img_array, depth)
            atlas_sub_arrays = split_into_subarrays(atlasDict[scan_name], depth)  # Assuming corresponding atlas exists
            
            for sub_img_array, atlas_sub_array in zip(target_sub_arrays, atlas_sub_arrays):
                sub_boundary_array = find_boundary(sub_img_array)
                
                combined_slices = np.stack((atlas_sub_array, sub_img_array), axis=-1)
                combined_slices_exp = np.expand_dims(combined_slices, axis=0)
                sub_boundary_array_exp = np.expand_dims(np.expand_dims(sub_boundary_array, axis=0), axis=-1)
                
                loss = model.train_on_batch(combined_slices_exp, sub_boundary_array_exp)
                loss_list.append(loss)
                
                prediction = model.predict(combined_slices_exp)
                prediction = (prediction[0, :, :, :, 0] > 0.5).astype(np.uint8)
                
                slice_index = depth // 2
                accumulated_slices.append((sub_img_array[slice_index], prediction[slice_index]))
                
                if len(accumulated_slices) == 3:
                    show_slices(accumulated_slices)
                    
                    while True:
                        feedback = input("Is this batch acceptable? (y/n): ")
                        
                        if feedback.lower() == 'y':
                            accumulated_slices = []
                            break
                            
                        elif feedback.lower() == 'n':
                            retrained_slices = []
                            
                            for original_slice, _ in accumulated_slices:
                                surrounding_slices = get_surrounding_slices(original_slice, target_sub_arrays, slice_index, depth)
                                atlas_slices = get_surrounding_slices(original_slice, atlas_sub_arrays, slice_index, depth)
                                
                                combined_slices = np.stack((atlas_slices, surrounding_slices), axis=-1)
                                combined_slices_exp = np.expand_dims(combined_slices, axis=0)
                                sub_boundary_array = find_boundary(surrounding_slices)
                                sub_boundary_array_exp = np.expand_dims(np.expand_dims(sub_boundary_array, axis=0), axis=-1)
                                
                                model.train_on_batch(combined_slices_exp, sub_boundary_array_exp)
                                
                                new_prediction = model.predict(combined_slices_exp)
                                new_prediction = (new_prediction[0, :, :, :, 0] > 0.5).astype(np.uint8)
                                new_slice_index = depth // 2
                                retrained_slices.append((surrounding_slices[new_slice_index], new_prediction[new_slice_index]))
                            
                            accumulated_slices = retrained_slices
                            show_slices(accumulated_slices)
                            
        proceed = input("Would you like to proceed to the next epoch? (y/n): ")
        if proceed.lower() != 'y':
            break    

    model.save('my_model.keras')
    loaded_model = load_model('my_model.keras')
    loaded_model.summary()

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
