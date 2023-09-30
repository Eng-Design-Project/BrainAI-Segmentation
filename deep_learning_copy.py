import SimpleITK as sitk
import os
import tensorflow as tf
import numpy as np
from scipy.ndimage import convolve
import data


def print_hello():
    print(" Entered deep learning ")


# New U-Net architecture
def unet(pretrained_weights=None, input_size=(256, 256, 256, 1)):
    inputs = tf.keras.layers.Input(input_size)
    conv1 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(inputs)
    pool1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    # ... More layers here ...
    
    # Final layer
    flatten = tf.keras.layers.Flatten()(pool1)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(flatten)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    if pretrained_weights:
        model.load_weights(pretrained_weights)
        
    return model



#dummy input data
sitk_images_dict = {
    "image1": data.get_3d_image("scan1"),
    "image2": data.get_3d_image("scan2"),   
    # Add other images...
}

#normalizes pixel value of 3d array
def normalizeTF(volume3dDict):
    normalizedDict = {}
    for key, value in volume3dDict.items():
        tensor = tf.convert_to_tensor(value, dtype=tf.float32)
        minVal = tf.reduce_min(tensor)
        maxVal = tf.reduce_max(tensor)
        normalizedTensor = (tensor - minVal) / (maxVal - minVal)
        
        # Convert back to numpy and store it in the dictionary
        normalizedDict[key] = normalizedTensor.numpy()
    return normalizedDict


#standard binary classifier, probably not useful for our use-case
def buildModel(inputShape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=inputShape),  # Corrected here
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  # Assumes binary classification
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

#pix by pix classifier, not built with user score in mind
def buildPixelModel(input_shape, window_size=3):
    # Assumes input is a 3D patch of size [window_size, window_size, depth]
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (window_size, window_size), activation='relu', padding='valid'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  # Assumes binary classification
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model        

#wrapper for getting the np arrays from sitkimages, normalizing, getting shape,
# and plugging shape into basic classifier
def dlAlgorithm(segmentDict):
    numpyImagesDict = {key: sitk.GetArrayFromImage(img) for key, img in segmentDict.items()}
    normalizedDict = normalizeTF(numpyImagesDict)

    """Currently using 3D arrays, might switch to tensors. In such case, the shape might change."""
    sampleShape = numpyImagesDict[list(numpyImagesDict.keys())[0]].shape
    model = buildModel((sampleShape[1], sampleShape[2], sampleShape[0]))  # (height, width, channels)

#finds edges of the image, only need to classify edges, not the entire thing
def find_boundary(segment):
    # Define a kernel for 3D convolution that checks for 26 neighbors in 3D
    kernel = np.ones((3, 3, 3))
    kernel[1, 1, 1] = 0
    
    # Convolve with the segment to find the boundary of ROI
    boundary = convolve(segment > 0, kernel) > 0
    
    # Keep only boundary voxels that are non-zero in the original segment
    boundary = boundary & (segment > 0)
    
    return boundary

#takes boundary (edges), and gets 3d windows around each boundary voxel. These are inputs to the model
def extract_windows(volume, boundary, window_size=3):
    padding = window_size // 2
    padded_volume = np.pad(volume, ((padding, padding), (padding, padding), (padding, padding)), mode='constant')
    windows = []
    indices = []
    
    boundary_indices = np.argwhere(boundary)  # Find the indices of boundary voxels
    
    for index in boundary_indices:
        z, y, x = index
        window = padded_volume[z:z + window_size, y:y + window_size, x:x + window_size]
        windows.append(window)
        indices.append((z, y, x))
    
    return np.array(windows), np.array(indices)

def build_boundary_window_model(window_size=3):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', input_shape=(window_size, window_size, window_size, 1)),
        tf.keras.layers.MaxPooling3D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  # Assuming binary classification (0 or 1)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Modified training function
def train_model(model, windows, labels, success_metric):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(windows, labels, epochs=int(success_metric * 10))

def classify_voxels(segment_volume, success_metric, window_size=3):
    boundary = find_boundary(segment_volume)
    windows, indices = extract_windows(segment_volume, boundary, window_size)
    windows = windows[..., np.newaxis]  # Adding a channel dimension

    # Here, I'm using your U-Net model instead of a newly built one.
    model = unet(input_size=(window_size, window_size, window_size, 1))

    # Dummy labels for demonstration; replace with actual labels if available
    labels = np.random.randint(0, 2, size=len(windows))
    
    # Train the model
    train_model(model, windows, labels, success_metric)
    
    predictions = model.predict(windows)
    predicted_labels = np.argmax(predictions, axis=1)
    
    classified_indices = indices[predicted_labels == 1]
    return classified_indices.tolist()


def test_classify_voxels():
    boundary_volume = np.random.randint(0, 2, (128, 128, 128))  # Replace with your actual boundary volume
    success_metric = 0.8  # Replace with your actual success metric
    classified_indices = classify_voxels(boundary_volume, success_metric)
    print(classified_indices)

if __name__ == '__main__':
    test_classify_voxels()

