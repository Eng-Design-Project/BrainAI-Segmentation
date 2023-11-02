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

#since our classifier model is looking at windows, labeled data must be in the form of windows
#i.e. a dictionary, key is brain region, value is a 4d np array. 3 dimensions are a single window 
# (a 3x3x3 chunk of the original image around the boundary)
#4th dimension is the index of that window
def generate_dummy_labeled_data(windows_dict):
    labeled_data = {}
    for region, windows in windows_dict.items():
        num_windows = windows.shape[0]
        # Randomly generate labels (0 or 1) for each window
        labels = np.random.randint(0, 2, num_windows)
        labeled_data[region] = labels
    return labeled_data

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

#this will break without labeled data
#this is deprecated, the logic is all in the class definition now
def train_model(model, windows, user_score, labels=None):
    model.fit(windows, labels, epochs= 20 - int(user_score * 10)) 

#this should run without labeled data
def train_model_with_user_feedback(model, windows, user_score, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(windows)[:, 1]
        loss = tf.keras.losses.MSE(tf.constant([user_score], dtype=tf.float32), tf.reduce_mean(predictions))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# the function below will not be used: the class method will be used instead
#only keeping this for now as a note
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
#notes: this class assumes one model can be used for all regions to be classified.
#  Probably not the case, so another class will be made that has a seperate model for each region
class CustomClassifierSingleModel:
    def __init__(self, initial_model=None):
        self.model = initial_model if initial_model else buildPixelModel()
        #self.models = {"brain": buildPixelModel()}
        self.classification_dict = {}
        self.normalized_data = None
        self.labeled_data = None
        self.windows_dict = {}
        self.optimizer = tf.keras.optimizers.Adam()  # Initialize once at the class level

    def executeDL(self, user_score=0, dict_of_np_arrays=None, labeled_data=None):

        # initialize data on first run of executeDL
        if self.normalized_data is None:
            self.normalized_data = normalize_np_dict(dict_of_np_arrays)

        # update labeled_data if it's given
        if labeled_data:
            self.labeled_data = labeled_data

        for region, seg_volume in self.normalized_data.items():
            if region not in self.windows_dict:
                windows, indices = extract_windows(seg_volume)
                self.windows_dict[region] = windows[..., np.newaxis]  # Adding a channel dimension
            else:
                windows = self.windows_dict[region]

            # Using the labeled_data for the specific region if available
            region_labels = self.labeled_data.get(region) if self.labeled_data else None

            # No need to compile multiple times, so we check if it's compiled.
            if not hasattr(self.model, 'optimizer'):
                self.model.compile(optimizer=self.optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

            if region_labels:
                self.model.fit(windows, region_labels, epochs=10)  # or any other number of epochs
                #self.models[region].fit(....)
            train_model_with_user_feedback(self.model, windows, user_score, self.optimizer)

            predictions = self.model.predict(windows)
            predicted_labels = np.argmax(predictions, axis=1)

            # Filter out the indices where the prediction is positive
            classified_indices = indices[predicted_labels == 1].tolist()
            self.classification_dict[region] = classified_indices

        return self.classification_dict

#temp putting this function here while md finalises pydicom conversion
def subfolders_to_3d_array_dictionary(directory):
    region_dict = {}
    for i in os.listdir(directory):
        region_dict[i] = data.get_3d_array_from_file(os.path.join(directory, i))
    return region_dict

if __name__ == '__main__':
   print("running dl module")
   classifier = CustomClassifierSingleModel()
   #need dict of np arrays
   test_data_input = {"scan1": data.get_3d_array_from_file("scan1"), "scan2": data.get_3d_array_from_file("scan2"),}
   classif_dict = classifier.executeDL(0, test_data_input)
   for keys, values in classif_dict.items():
       print(keys, ": ", values)
   

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

