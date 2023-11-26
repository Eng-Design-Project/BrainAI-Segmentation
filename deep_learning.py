import os
import tensorflow as tf
import numpy as np
from scipy.ndimage import convolve
import data
import segmentation


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


#hoping to give the model a starting point to work with,
#  so it can make predictions at the start that are at least passable
def generate_heuristic_labeled_data(windows_dict, threshold=0.3):
    labeled_data = {}
    for region, windows in windows_dict.items():
        labels = []
        for window in windows:
            #calculate the proportion of voxels in the window that are above the threshold
            prop_above_threshold = np.mean(window > threshold)
            #if the majority of voxels are above the threshold, label as 0, else 1
            label = 0 if prop_above_threshold > 0.5 else 1
            labels.append(label)
        labeled_data[region] = np.array(labels)
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
def buildPixelModel(window_size=8):
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
def extract_windows(volume, window_size=8):
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

#this should run without labeled data
def train_model_with_user_feedback(model, windows, user_score, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(windows)[:, 1]
        loss = tf.keras.losses.MSE(tf.constant([user_score], dtype=tf.float32), tf.reduce_mean(predictions))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

    
class CustomClassifierMultiModel:
    def __init__(self, dict_of_np_arrays):
        self.normalized_data = normalize_np_dict(dict_of_np_arrays)
        self.model_dict = {region: buildPixelModel() for region in self.normalized_data.keys()}
        self.optimizer_dict = {region: tf.keras.optimizers.Adam() for region in self.normalized_data.keys()}
        self.windows_dict = {}
        self.indices_dict = {}
        
        #extract windows from input data
        for region, seg_volume in self.normalized_data.items():
            windows, indices = extract_windows(seg_volume)
            self.windows_dict[region] = windows[..., np.newaxis]
            self.indices_dict[region] = indices
        
        self.labeled_data = generate_heuristic_labeled_data(self.windows_dict)

        for region, seg_volume in self.normalized_data.items():
            model = self.model_dict[region]
            optimizer = self.optimizer_dict[region]
            region_labels = self.labeled_data[region]
            model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            model.fit(self.windows_dict[region], region_labels, epochs=10)

    def trainDL(self, user_score=0):
        classification_dict = {}
        for region, windows in self.windows_dict.items():
            model = self.model_dict[region]
            optimizer = self.optimizer_dict[region]

            # User feedback training
            if windows.size == 0:
                print(f"No windows to process for region: {region}")
                continue

            train_model_with_user_feedback(model, windows, user_score, optimizer)

            # Predictions
            predictions = model.predict(windows)
            predicted_labels = np.argmax(predictions, axis=1)
            classified_indices = self.indices_dict[predicted_labels == 1].tolist()
            classification_dict[region] = classified_indices

        return classification_dict
    
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for region, model in self.model_dict.items():
            model_path = os.path.join(save_dir, f'model_{region}.h5')
            model.save(model_path)
        print(f"All models saved in {save_dir}")
    
    def load_models(self, save_dir):
        for region in self.normalized_data.keys():
            model_path = os.path.join(save_dir, f'model_{region}.h5')
            if os.path.exists(model_path):
                self.model_dict[region] = tf.keras.models.load_model(model_path)
            else:
                print(f"No saved model found for region {region} in {save_dir}")
    
    def predict(self, dict_of_np_arrays):
        predictions_dict = {}
        for region, seg_volume in dict_of_np_arrays.items():
            if region in self.model_dict:
                windows, _ = extract_windows(seg_volume)
                windows = windows[..., np.newaxis]
                model = self.model_dict[region]
                predictions = model.predict(windows)
                predicted_labels = np.argmax(predictions, axis=1)
                predictions_dict[region] = predicted_labels
            else:
                print(f"No model found for region {region}")
        return predictions_dict

if __name__ == '__main__':
    print("running dl module")
    # classifier = CustomClassifierSingleModel()
    
    #need dict of np arrays
    test_data_input = data.subfolders_to_dictionary("scan1 atl seg.DCMs")
    for key in test_data_input.keys():
        print(key + " shape:")
        print(test_data_input[key].shape)

    if (test_data_input != None):
        del test_data_input["Skull"]

        classifier = CustomClassifierMultiModel(test_data_input)
        classif_dict = classifier.trainDL()
        # for keys, values in classif_dict.items():
        #     print(keys, ": ", values)
        results = segmentation.filter_noise_from_images(test_data_input, classif_dict)
        data.display_seg_np_images(results)
    

    #need to put execution in for loop, get user_score each time
    #need to generate some labeled data: handmake a brain with neat boundary, extract windows, label all windows 1
    #or use atlas and make a heuristic to generate labeled windows: if >30% of window > 0, 1, else 0
    #need to save model between runs somehow
   

'''

#for the dummyDL function
#labeled data is tricky, because it should probably be in the same format as the input data + a 1 or 0 label
#dictionary makes sense, but the input data is a 3x3x3 (unless we pass a window size other than default) np array,
# or rather, a stack of 3x3x3 arrays inside another array, because that's more optimized
dummyLabels = {
    "image1": 1,
    "image2": 0
    # Add more labels...
}


#this will break without labeled data
#this is deprecated, the logic is all in the class definition now
def train_model(model, windows, user_score, labels=None):
    model.fit(windows, labels, epochs= 20 - int(user_score * 10)) 

    
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

'''
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
                for i, window in enumerate(windows[:15]):  # Print first 5 windows
                    print(f"Window {i}: shape = {window.shape}")
                    print("First slice of the window:")
                    print(window[0, :, :])  # This will print only the first slice of the 3D window
                self.windows_dict[region] = windows[..., np.newaxis]  # Adding a channel dimension
                print("region not in windows dict")
            else:
                windows = self.windows_dict[region]
                print("region in windows dict")

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
            classified_indices = indices[predicted_labels == 0].tolist()
            self.classification_dict[region] = classified_indices

        return self.classification_dict

class CustomClassifierMultiModelold:
    def __init__(self, regions=None):
        self.model_dict = {region: buildPixelModel() for region in regions} if regions else {}
        self.classification_dict = {}
        self.normalized_data = None
        self.labeled_data = None
        self.windows_dict = {}
        self.optimizer_dict = {region: tf.keras.optimizers.Adam() for region in regions} if regions else {}

    def trainDL(self, user_score=0, dict_of_np_arrays=None, labeled_data=None):
        # Initialize data and models
        if self.normalized_data is None:
            self.normalized_data = normalize_np_dict(dict_of_np_arrays)
            for region in self.normalized_data.keys():
                if region not in self.model_dict:
                    self.model_dict[region] = buildPixelModel()
                    self.optimizer_dict[region] = tf.keras.optimizers.Adam()

        # Update labeled data if provided
        if labeled_data:
            self.labeled_data = labeled_data

        for region, seg_volume in self.normalized_data.items():
            # Extract windows
            if region not in self.windows_dict:
                windows, indices = extract_windows(seg_volume)
                self.windows_dict[region] = windows[..., np.newaxis]
            else:
                windows = self.windows_dict[region]

            model = self.model_dict[region]
            optimizer = self.optimizer_dict[region]
            region_labels = self.labeled_data.get(region) if self.labeled_data else None

            # Training
            if region_labels:
                model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                model.fit(windows, region_labels, epochs=10)
            else:
                train_model_with_user_feedback(model, windows, user_score, optimizer)

            # Prediction
            predictions = model.predict(windows)
            predicted_labels = np.argmax(predictions, axis=1)
            classified_indices = indices[predicted_labels == 0].tolist()
            self.classification_dict[region] = classified_indices

        return self.classification_dict
    
    '''

