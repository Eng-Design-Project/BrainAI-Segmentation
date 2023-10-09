# Import relevant modules
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
def unet(input_size=(5, 128, 128, 1)):
    
    inputs = tf.keras.layers.Input(input_size)# Define input layer
    
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


def dlAlgorithm(segmentDict, depth=5, epochs=3):
    numpyImagesDict = {key: sitk.GetArrayFromImage(img) for key, img in segmentDict.items()} # Convert SimpleITK images to NumPy arrays
    normalizedDict = normalizeTF(numpyImagesDict) # Normalize the images
    model = unet(input_size=(depth, 128, 128, 1)) # Initialize U-Net model
    
    loss_list = [] # Loss list for tracking loss
    accumulated_slices = [] # List to accumulate slices for batch display
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")
        
        for img_array in normalizedDict.values():            
            sub_arrays = split_into_subarrays(img_array, depth)# Split into sub-arrays
             
            for sub_img_array in sub_arrays: # Iterate over sub-arrays for training
                sub_boundary_array = find_boundary(sub_img_array) # Find the boundary segmentation
                
                # Expand dimensions to make it compatible for training
                sub_img_array_exp = np.expand_dims(np.expand_dims(sub_img_array, axis=0), axis=-1)
                sub_boundary_array_exp = np.expand_dims(np.expand_dims(sub_boundary_array, axis=0), axis=-1)
                
                 # Train the model on the current batch
                loss = model.train_on_batch(sub_img_array_exp, sub_boundary_array_exp)
                loss_list.append(loss)
                
                # Predict the segmentation
                prediction = model.predict(sub_img_array_exp)
                prediction = (prediction[0, :, :, :, 0] > 0.5).astype(np.uint8)
                
                slice_index = depth // 2  # Take a middle slice to visualize
                
                # Append as tuple (original_slice, prediction_slice)
                accumulated_slices.append((sub_img_array[slice_index], prediction[slice_index]))
                
                # Check if 3 batches have been accumulated, and if so, show them
                if len(accumulated_slices) >= 3:
                    show_slices(accumulated_slices)
                    accumulated_slices = []  # Clear the accumulated_slices for the next set

                    # Ask for user feedback
                    feedback = input("Is this batch acceptable? (y/n): ")
                    
                    if feedback.lower() == 'n':  # Retrain on the same batch
                        for sub_img_array, _ in accumulated_slices:
                            sub_boundary_array = find_boundary(sub_img_array)
                            sub_img_array_exp = np.expand_dims(np.expand_dims(sub_img_array, axis=0), axis=-1)
                            sub_boundary_array_exp = np.expand_dims(np.expand_dims(sub_boundary_array, axis=0), axis=-1)
                            model.train_on_batch(sub_img_array_exp, sub_boundary_array_exp)
        # Prompt for continuing to next epoch
        proceed = input("Would you like to proceed to the next epoch? (y/n): ")
        if proceed.lower() != 'y':
            break    
                    
    model.save('my_model.keras')
    loaded_model = load_model('my_model.keras')
    loaded_model.summary()

    # Show the loss curve
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





