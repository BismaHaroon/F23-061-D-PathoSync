
import os

from tensorflow.keras import layers
from tensorflow.keras import Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split
import random
# preprocessing
import shutil
import sys


# ###epochs and parameters
# #Extract epochs, learning rate, and batch size from command-line arguments
# Epochs = None
# learning_rate = None
# batch_size = None
# for i in range(len(sys.argv)):
#     if sys.argv[i] == "--epochs" and i + 1 < len(sys.argv):
#         Epochs = int(sys.argv[i + 1])
#     elif sys.argv[i] == "--learning_rate" and i + 1 < len(sys.argv):
#         learning_rate = float(sys.argv[i + 1])
#     elif sys.argv[i] == "--batch_size" and i + 1 < len(sys.argv):
#         Batch_size = int(sys.argv[i + 1])

# # Check if any parameter is missing
# if Epochs is None or learning_rate is None or Batch_size is None:
#     print("Error: One or more parameters are missing.")
#     sys.exit(1)

# # Now you can use the 'epochs', 'learning_rate', and 'batch_size' variables in your model training code
# print(f"Epochs: {Epochs}, Learning Rate: {learning_rate}, Batch Size: {Batch_size}")
# Define a custom callback to print epoch information

class EpochInfoCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}/{self.params['epochs']}:")
        sys.stdout.flush()
        print(f"  - Training Loss: {logs['loss']}, Accuracy: {logs['acc']}")
        sys.stdout.flush()
        print(f"  - Validation Loss: {logs['val_loss']}, Accuracy: {logs['val_acc']}")
        sys.stdout.flush()

# Define the function to flush standard output
def flush_stdout():
    sys.stdout.flush()

# Define the function to extract class names from file names
def extract_class_name(file_name):
    return file_name.split('_')[0]  # Assuming class name is before the first underscore

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


try:
    # Filter out only directories
    directories = [f for f in os.listdir(uploads_folder) if os.path.isdir(os.path.join(uploads_folder, f))]
    
    # Find the  appropriate directory
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

# Create the joint folder for the dataset
dataset_folder = f"{dataset_name}_dataset"
dataset_path = os.path.join(script_dir, dataset_folder)
os.makedirs(dataset_path, exist_ok=True)
print(f"Created joint dataset folder: {dataset_path}")
flush_stdout()
# Traverse all subdirectories within the obtained directory
for root, dirs, files in os.walk(os.path.join(uploads_folder, most_recent_dir)):
    for file in files:
        file_path = os.path.join(root, file)
        # Copy the file to the joint dataset folder
        shutil.copy(file_path, dataset_path)
        # print(f"Saved file '{file}' in joint dataset folder")

print("Preprocessing completed.","\n")
flush_stdout()
print('Proceeding to Train/Test Split', '\n')
flush_stdout()
# Remove existing train and test directories if they exist
train_dir = os.path.join(dataset_path, 'train')
test_dir = os.path.join(dataset_path, 'test')

if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)


# Lists to store file paths for train and test sets
train_files = []
test_files = []

# Traverse all subdirectories within the obtained directory
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        file_path = os.path.join(root, file)
        if file.endswith(".jpg") or file.endswith(".png"):  # Considering only image files
            if random.random() < 0.6:  # 60% chance of adding to train set
                train_files.append(file_path)
            else:
                test_files.append(file_path)

# Define train and test directories within the dataset directory
train_dir = os.path.join(dataset_path, 'train')
test_dir = os.path.join(dataset_path, 'test')

# Create train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Organize files into subdirectories based on class names
for file_path in train_files:
    class_name = extract_class_name(os.path.basename(file_path))
    class_dir = os.path.join(train_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    shutil.copy(file_path, class_dir)


for file_path in test_files:
    class_name = extract_class_name(os.path.basename(file_path))
    class_dir = os.path.join(test_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    shutil.copy(file_path, class_dir)

print("Train/Test split completed successfully.")
flush_stdout()
pre_trained_model_2 = DenseNet201(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3))

# pre_trained_model_2.summary()

print(f"<><><><> Train Dir: '{train_dir}'<><><><>")
flush_stdout()
print(f"<><><><> Test Dir: '{test_dir}'<><><><>")
flush_stdout()

x = layers.GlobalAveragePooling2D()(pre_trained_model_2.output)
x = layers.Dropout(0.2)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense  (1, activation='sigmoid')(x)

model = Model( pre_trained_model_2.input, x) 
# augmentation
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


test_datagen = ImageDataGenerator( rescale = 1.0/255. )


# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 10,
                                                    class_mode = 'binary',
                                                    target_size = (224, 224))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory( test_dir,
                                                          batch_size  = 10,
                                                          class_mode  = 'binary',
                                                          target_size = (224, 224))

contents_folder = os.path.join(dataset_path, "contents")
os.makedirs(contents_folder, exist_ok=True)
print("Contents folder path:", contents_folder)



# Define the path for saving the model
model_path = os.path.join("trained_models_tissue", f"{dataset_name}_ConVNet.h5")
checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# Define and compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='binary_crossentropy', 
              metrics=['acc'])

# Train the model with the custom callback
history = model.fit(
            train_generator,
            validation_data=validation_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=1,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            verbose=1,
            callbacks=[checkpoint, EpochInfoCallback()])

# Save the model as a single .h5 file
model.save(model_path)