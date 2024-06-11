# Import Modules
import os
import splitfolders
# Deep Learning
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
import sys
import shutil 

if os.path.exists('output'):
    shutil.rmtree('output')

# Define the function to flush standard output
def flush_stdout():
    sys.stdout.flush()

print("Script started executing...")
flush_stdout()
# Get the absolute path to the directory where this script resides
script_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the uploads directory
uploads_folder = os.path.join(script_dir, 'uploads')

# Check if the uploads directory exists
if not os.path.exists(uploads_folder):
    print("Error: 'uploads' directory not found")
    flush_stdout()
    # Handle the error gracefully, e.g., by exiting the script or raising an exception
else:
    print("Uploads folder found successfully.")
    flush_stdout()

# Get the most recently created directory in the uploads folder
try:
    # Filter out only directories
    directories = [f for f in os.listdir(uploads_folder) if os.path.isdir(os.path.join(uploads_folder, f))]
    
    # Find the most recently created directory
    most_recent_dir = max(directories, key=lambda f: os.path.getctime(os.path.join(uploads_folder, f)))
    dataset_name = most_recent_dir
    print(f"Dataset '{dataset_name}' ")
    flush_stdout()
except ValueError:
    print("Error: No directories found in 'uploads' folder")
    flush_stdout()
    # Handle the error gracefully, e.g., by exiting the script or raising an exception
except Exception as e:
    print(f"Error: {e}")
    flush_stdout()

dataset_path = f"uploads/{dataset_name}"
# Count the number of subdirectories (classes) in the dataset
num_of_classes = len([name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))])

# Print the number of classes for debugging
print("Number of classes:", num_of_classes)

input_ = dataset_path

# split data into training, vlaidation sets
splitfolders.ratio(input_, 'output', seed = 101, ratio=(0.8, 0.1, 0.1))

data_dir = 'output'

# Define train, valid and test directories
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'val')

# Print out directory paths for debugging
print("Train directory:", train_dir)
print("Validation directory:", valid_dir)

os.listdir('output')

# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255.,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   shear_range=0.4,
                                   width_shift_range=0.25,
                                   height_shift_range=0.25,
                                   rotation_range=45,
                                   fill_mode='nearest')
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Train ImageDataGenerator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 64,
                                                    target_size = (224,224),
                                                    class_mode = 'categorical',
                                                    shuffle=True,
                                                    seed=42,
                                                    color_mode='rgb')

valid_generator = valid_datagen.flow_from_directory(valid_dir,
                                                    batch_size=64,
                                                    target_size=(224,224),
                                                    class_mode='categorical',
                                                    shuffle=True,
                                                    seed=42,
                                                    color_mode='rgb')

base_model_resnet50 = ResNet50(input_shape=(224,224,3),
                               include_top=False, 
                               weights='imagenet')

x = base_model_resnet50.output
flat = Flatten()(x)
hidden_1 = Dense(1024, activation='relu')(flat)
drop_1 = Dropout(0.2)(hidden_1)
hidden_2 = Dense(512, activation='relu')(drop_1)
drop_2 = Dropout(0.3)(hidden_2)
output = Dense(num_of_classes, activation= 'softmax')(drop_2)

model_resnet50_01 = Model(base_model_resnet50.inputs, output)

# Call Backs
filepath = os.path.join("trained_models_tissue", f"{dataset_name}_ResNet.h5")
es = EarlyStopping(monitor='loss', verbose=1, mode='min', patience=4)
cp = ModelCheckpoint(filepath, monitor='val_loss',verbose=1, save_best_only=True,
                     save_weights_only=False, mode='auto', period=1)
# compile the model
num_epochs=10
model_resnet50_01.compile(loss="categorical_crossentropy", optimizer=Adam(lr = 0.0004), metrics=["accuracy"])
resnet50_history_01 = model_resnet50_01.fit_generator(train_generator,
                                                      steps_per_epoch=10,
                                                      epochs=num_epochs,
                                                      callbacks = [es, cp],
                                                      validation_data = valid_generator)