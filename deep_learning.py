import SimpleITK as sitk
import os
import tensorflow as tf
import numpy as np
import data  # Make sure you have this data module

def normalizeTF(volume3dDict):
    normalizedDict = {}
    for key, value in volume3dDict.items():
        tensor = tf.convert_to_tensor(value, dtype=tf.float32)
        minVal = tf.reduce_min(tensor)
        maxVal = tf.reduce_max(tensor)
        normalizedTensor = (tensor - minVal) / (maxVal - minVal)
        normalizedDict[key] = normalizedTensor.numpy()
    return normalizedDict

def buildModel(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),  # Corrected here
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

    def normalize_data(self, volume3dDict):
        return normalizeTF(volume3dDict)

    def build_and_train_model(self):
        # Assume you've normalized and prepared your data, and it's stored in self.normalized_data
        sample_shape = list(self.normalized_data.values())[0].shape
        model = buildModel((sample_shape[1], sample_shape[2], sample_shape[0]))



        return model

    def evaluate_model(self, model, X_test, y_test):
        loss, acc = model.evaluate(X_test, y_test)
        return loss, acc

    def set_user_score(self, x1, x2):
        self.user_score1 = x1
        self.user_score2 = x2
        print(f"score 1 is: {self.user_score1}")
        print(f"score 2 is: {self.user_score2}")

# Example usage:

dl_module = DeepLearningModule()
dl_module.load_regions(region_data={})  # Fill in the region_data dictionary
dl_module.load_atlas_data("scan1", "scan2")  # Replace with actual paths

# Normalize data
volume3dDict = {}  # Assume you fill in this dictionary
normalized_data = dl_module.normalize_data(volume3dDict)

# Build, train, and evaluate model
inputShape = (128, 128, 3)  # Replace with the actual shape of your data
X_train = np.array([])  # Replace with your actual training data
y_train = np.array([])  # Replace with your actual labels
X_test = np.array([])  # Replace with your actual test data
y_test = np.array([])  # Replace with your actual test labels

model = dl_module.build_and_train_model(inputShape, X_train, y_train)
loss, acc = dl_module.evaluate_model(model, X_test, y_test)

# Set user scores
dl_module.set_user_score(5, 6)
