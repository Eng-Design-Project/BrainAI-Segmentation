import SimpleITK as sitk
import os
import tensorflow as tf
import numpy as np
from scipy.ndimage import convolve
import data

#dummy input data
sitk_images_dict = {
    "image1": data.get_3d_image("scan1"), # gets 3d sitk image from a folder of DCM images
    "image2": data.get_3d_image("scan2"),   
    # Add other images...
}

#for the dummyDL functiom
numpyImagesDict = {key: sitk.GetArrayFromImage(img) for key, img in sitk_images_dict.items()}

#for the dummyDL function
#labeled data is tricky, because it should probably be in the same format as the input data + a 1 or 0 label
#dictionary makes sense, but the input data is a 3x3x3 (unless we pass a window size other than default) np array,
# or rather, a stack of 3x3x3 arrays inside another array, because that's more optimized
dummyLabels = {
    "image1": 1,
    "image2": 0
    # Add more labels...
}

#normalizes pixel value of 3d array dict
def normalize_np_dict(volume3dDict):
    normalizedDict = {}
    for key, value in volume3dDict.items():
        tensor = tf.convert_to_tensor(value, dtype=tf.float32)
        minVal = tf.reduce_min(tensor)
        maxVal = tf.reduce_max(tensor)
        normalizedTensor = (tensor - minVal) / (maxVal - minVal)
        
        # Convert back to numpy and store it in the dictionary
        normalizedDict[key] = normalizedTensor.numpy()

        #note: it would be more optimized to normalize each input window rather than 
        # normalizing the entire image. Prob doesn't matter
    return normalizedDict


#pix by pix classifier, not built with user score in mind
def buildPixelModel(window_size=3):
    # Assumes input is a 3D patch of size [window_size, window_size, window_size]
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(window_size, window_size, window_size, 1)),
        tf.keras.layers.Conv3D(32, (window_size, window_size, window_size), activation='relu', padding='valid'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  # Assumes binary classification
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model          



#finds edges of the image, only need to classify edges, not the entire thing
def find_boundary(segment):
    # Define a kernel for 3D convolution that checks for 26 neighbors in 3D
    kernel = np.ones((3, 3, 3))
    kernel[1, 1, 1] = 0
    
    # Convolve with the segment to find the boundary of ROI
    boundary = convolve(segment > 0, kernel) > 0
    
    return boundary

#takes boundary (edges), and gets 3d windows around each boundary voxel. These are inputs to the model
def extract_windows(volume, window_size=3):
    boundary = find_boundary(volume)
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
    
    #does it make sense to convert to arrays?
    return np.array(windows), np.array(indices)


def train_model(model, windows, user_score, labels=None):
    model.fit(windows, labels, epochs= 20 - int(user_score * 10)) 

# not tested yet -Kevin
def executeDL(dict_of_np_arrays, user_score=0, model=buildPixelModel()):
    
    #should only have to normalize data once, and we only need to pass dict_of_np_arrays once
    normalized_data = normalize_np_dict(dict_of_np_arrays) 

    #should only have to compile once?
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics = ["accuracy"])
    
    #train model on on dict of np arrays,
        # Dummy labels for demonstration; replace with actual labels if available
    #need to run for each key/value?
    classification_dict = {}
    for region, seg_volume in normalized_data.items():
        windows, indices = extract_windows(seg_volume)
        windows = windows[..., np.newaxis]  # Adding a channel dimension
        
        train_model(model, windows, dummyLabels, user_score)
        
        predictions = model.predict(windows)
        predicted_labels = np.argmax(predictions, axis=1)
    
        classified_indices = indices[predicted_labels == 1]
        classification_dict[region] = classified_indices

    #return segmentation attempts (dictionary of coordinates)
    #return model
    return normalized_data, model, classification_dict

    #segmentation attemps displayed by core, user score collected
    #this function called again, passing the model back, and passing in user score
    #dummyDL is run on a loop, probably in core?

#class version of executeDL: no need to output and reinput things other than user_score
#Notes: labeled data could be a bunch of 'windows', labeled 0 or 1
#   should windows also be a class attribute? why extract windows many times
#   if windows are a class attribute, than we wouldn't need to keep the entire image in memory
class CustomClassifier:
    def __init__(self, initial_model=None, labeled_data=None):
        self.model = initial_model if initial_model else self.buildPixelModel()
        self.classification_dict = {}
        self.normalized_data = None
        self.labeled_data = labeled_data


    def executeDL(self, dict_of_np_arrays, user_score=0):
        self.normalized_data = normalize_np_dict(dict_of_np_arrays)

        # No need to compile multiple times, so we check if it's compiled.
        if not hasattr(self.model, 'optimizer'):
            self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        for region, seg_volume in self.normalized_data.items():
            windows, indices = extract_windows(seg_volume)
            windows = windows[..., np.newaxis]  # Adding a channel dimension

            # Using the labeled_data for the specific region if available
            region_labels = self.labeled_data.get(region) if self.labeled_data else None
            #labeled data needs to be redone, to fit with the input data expected

            train_model(self.model, windows, user_score, region_labels)
            
            #make predictions
            predictions = self.model.predict(windows)

            #average predicted probability for positive class
            avg_prediction = np.mean(predictions[:, 1])

            #what datastruct is predictions? does this round or just make things 1? Find the highest preds?
            predicted_labels = np.argmax(predictions, axis=1)

            classified_indices = indices[predicted_labels == 1]
            self.classification_dict[region] = classified_indices

        return self.classification_dict
    
def train_with_user_feedback(self, windows, user_score):
        # Step 1: Predict with current model
        predictions = self.model.predict(windows)
        
        # Step 2: Compute average predicted probability for positive class
        avg_prediction = np.mean(predictions[:, 1])
        
        # Step 3: Define a loss based on the user's score
        # We can use mean squared error here, but other choices might be appropriate depending on the problem
        loss = (avg_prediction - user_score) ** 2
        
        # Step 4: Update the model
        # This part is tricky without labeled data; we need a way to compute gradients
        # One option is to use a library like TensorFlow's 'tape' mechanism
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            # Repredict to get the model's outputs as TensorFlow tensors
            predictions_tf = self.model(windows, training=True)
            avg_prediction_tf = tf.reduce_mean(predictions_tf[:, 1])
            loss_tf = (avg_prediction_tf - user_score) ** 2
        grads = tape.gradient(loss_tf, self.model.trainable_variables)
        optimizer = tf.keras.optimizers.Adam()
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


if __name__ == '__main__':
   print("running dl module")
   #classifier = CustomClassifier()
   #test_dir = "scan1"
   #test_input = data.get_3d_array_from_file(test_dir)
   
   #data.display_3d_array_slices(test_input, 20)

'''
#class not needed
class DeepLearningModule:
    def __init__(self):
        self.atlas_segmentation_data = {}
        self.user_score1 = -1
        self.user_score2 = -2

    def load_regions(self, region_data):
        for region_name, sitk_name in region_data.items():
            try:
                region_image = sitk.ReadImage(sitk_name)
                print(f"Loaded {region_name} from {sitk_name}")
            except Exception as e:
                print(f"Error loading {region_name} from {sitk_name}: {e}")

    def load_atlas_data(self, atlas_data1, atlas_data2):
        for folder_path in [atlas_data1, atlas_data2]:
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    atlas_image = sitk.ReadImage(file_path)
                    self.atlas_segmentation_data[filename] = atlas_image
                    print(f"Loaded atlas data from {file_path}")
                except Exception as e:
                    print(f"Error loading atlas data from {file_path}: {e}")


# Existing user score global variables and function
#will prob be removed, user score will be supplied to dl algo as argument from core
user_score1 = -1
user_score2 = -2

#will be removed, user score updated in core
def get_user_score(x1, x2):
    global user_score1, user_score2
    user_score1 = x1
    user_score2 = x2
    print("score 1 is: ", user_score1)
    print("score 2 is: ", user_score2)
'''


'''
#had two models that essentially did the same thing, 
#idk what the difference between them is. 
#moved the second here to avoid confusion
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
'''

