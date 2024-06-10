<<<<<<< HEAD
 ###added import to first line of file
from __future__ import annotations,print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from tifffile import imread

from csbdeep.utils import Path, normalize
from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D

import zipfile
import numpy as np
from skimage.io import imsave
import matplotlib.pyplot as plt
from models import UNet, NuClick_NN
import torch
import base64, uuid
import tempfile
import logging
from skimage.color import label2rgb
from skimage import img_as_ubyte 
import cv2
from PIL import Image
from datetime import datetime
import os
import glob
from config import DemoConfig as config
from utils.process import post_processing, gen_instance_map
os.add_dll_directory(r"D:\New folder\NuClick\Pathosync\Pathosync-backend\app\dll_files")
from tiatoolbox.tools.tissuemask import MorphologicalMasker
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.stainnorm import VahadaneNormalizer,ReinhardNormalizer
from tiatoolbox import data
from tiatoolbox.tools import patchextraction
from utils.misc import get_clickmap_boundingbox
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import supervision as sv
from utils.guiding_signals import get_patches_and_signals
import torch
from flask import Flask, request, render_template, send_from_directory
from flask_cors import CORS
from flask import jsonify
from skimage import io
from skimage import color
import skimage.io
from skimage import io, exposure, img_as_ubyte
from werkzeug.utils import secure_filename
from flask import url_for, send_file
from flask_caching import Cache
import tiatoolbox.utils.visualization as vis_utils
from flask_pymongo import PyMongo
from flask import Flask
from pymongo import MongoClient
from bson import ObjectId
import logging
import os
if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()
import shutil
import warnings
from pathlib import Path
from zipfile import ZipFile
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from tiatoolbox import logger
import tensorflow as tf
import glob
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tiatoolbox.models.engine.patch_predictor import (
    IOPatchPredictorConfig,
    PatchPredictor,
)
from tiatoolbox.utils.misc import download_data, grab_files_from_dir, imread
from tiatoolbox.utils.visualization import overlay_prediction_mask
from tiatoolbox.wsicore.wsireader import WSIReader

from pathlib import Path

from tiatoolbox.tools.tissuemask import MorphologicalMasker
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.stainnorm import VahadaneNormalizer,ReinhardNormalizer
from tiatoolbox import data
from tiatoolbox.tools import patchextraction
from tiatoolbox.models.engine.patch_predictor import (
    IOPatchPredictorConfig,
    PatchPredictor,
)
mpl.rcParams["figure.dpi"] = 160  # for high resolution figure in notebook
mpl.rcParams["figure.facecolor"] = "white"  # To make sure text is visible in dark mode
import matplotlib
matplotlib.use('Agg')

from stardist.models import StarDist2D
from stardist.plot import render_label
from csbdeep.utils import normalize
import torch
from stardist.plot import render_label
from csbdeep.utils import normalize
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

###imports for cell model training
# import tensorflow as tf
# import glob
# from pathlib import Path
# from skimage.io import imread
from skimage.transform import resize
#import numpy as np
#import random
#import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

ON_GPU = False

PATCH_PLOT_DIR = "patch_plots"
OVERLAY_PLOT_DIR = "overlay_plot"
training_status = {'status': 'idle', 'message': ''}

# Ensure directories exist
os.makedirs(PATCH_PLOT_DIR, exist_ok=True)
os.makedirs(OVERLAY_PLOT_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)
cache = Cache(app)


# Database Configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/PathoSync"
mongo = PyMongo(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def latest_processed():
    processed_images = []

    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.endswith("_normalized.png"):
                processed_images.append(filename)

    if processed_images:
        # Sort the processed images by modification time (assuming newer ones are at the end)
        processed_images.sort(key=lambda x: os.path.getmtime(os.path.join(app.config['UPLOAD_FOLDER'], x)))

        # Return the path to the latest processed image
        latest_processed_image = os.path.join(app.config['UPLOAD_FOLDER'], processed_images[-1])
        print("Latest Processed Image:", latest_processed_image)
        return latest_processed_image
    else:
        print("No processed images available")
        return None

# Functions to make image masks
def visualize_mask(image):

    pil_image = Image.open(image)
    image_np = np.array(pil_image)

    # Apply morphological dilation using OpenCV
    kernel = np.ones((5, 5), np.uint8)  # Define a kernel for dilation
    dilated_image = cv2.dilate(image_np, kernel)

    # Perform tissue masking using TIAToolbox
    masker = MorphologicalMasker()
    mask = masker.fit_transform([image_np])[0]

    # Convert the tissue mask to a PIL image
    mask_pil_image = Image.fromarray(mask)


    return mask_pil_image

def create_mask():
    image_path = latest_processed()
    mask_folder = 'mask'

    # Create a mask using visualize_mask function
    mask = visualize_mask(image_path)

    # Create the mask folder if it doesn't exist
    mask_directory = os.path.join(app.config['UPLOAD_FOLDER'], mask_folder)
    if not os.path.exists(mask_directory):
        os.makedirs(mask_directory)

    # Save the mask image with the desired filename in the 'mask' subdirectory
    mask_filename = os.path.join(mask_directory, os.path.basename(image_path).replace('_normalized.png', '_masked.png'))

    # Save the mask directly without converting to a PIL Image again
    mask.save(mask_filename)

    return mask_filename


# Functions for vahadane normalization VahadaneNormalizer
def vahadane_mask(image):
    vahad_vis= VahadaneNormalizer()
    pil_image = Image.open(image)
    img_array = np.array(pil_image)
    target_image = data.stain_norm_target()
    vahad_vis.fit(target_image)
    norm_img=vahad_vis.transform(img_array)
    return norm_img

def create_visual_vahadane():
    image_path = latest_processed()
    mask_folder = 'vahadane'

    # Create a mask using visualize_mask function
    visual = vahadane_mask(image_path)

    # Create the mask folder if it doesn't exist
    mask_directory = os.path.join(app.config['UPLOAD_FOLDER'], mask_folder)
    if not os.path.exists(mask_directory):
        os.makedirs(mask_directory)

    # Save the mask image with the desired filename in the 'mask' subdirectory
    mask_filename = os.path.join(mask_directory, os.path.basename(image_path).replace('_normalized.png', '_vahadane.png'))

    # Save the mask directly without converting to a PIL Image again
    visual_image = Image.fromarray((visual * 255).astype(np.uint8))
    visual_image.save(mask_filename)

    return mask_filename

# Functions for reinhard normalization ReinhardNormalizer
def reinhard_mask(image):
    vahad_vis= ReinhardNormalizer()
    pil_image = Image.open(image)
    img_array = np.array(pil_image)
    target_image=Image.open("target_image.png")
    vahad_vis.fit(target_image)
    norm_img=vahad_vis.transform(img_array)
    return norm_img

def create_visual_reinhard():
    image_path = latest_processed()
    mask_folder = 'reinhard'

    # Create a mask using visualize_mask function
    visual = reinhard_mask(image_path)

    # Create the mask folder if it doesn't exist
    mask_directory = os.path.join(app.config['UPLOAD_FOLDER'], mask_folder)
    if not os.path.exists(mask_directory):
        os.makedirs(mask_directory)

    # Save the mask image with the desired filename in the 'mask' subdirectory
    mask_filename = os.path.join(mask_directory, os.path.basename(image_path).replace('_normalized.png', '_reinhard.png'))

    # Save the mask directly without converting to a PIL Image again
    visual_image = Image.fromarray((visual * 255).astype(np.uint8))
    visual_image.save(mask_filename)

    return mask_filename

def create_mask_nuclick():
    image_path = latest_processed()
    mask_folder = 'mask'

    # Create a mask using visualize_mask function
    mask = visualize_mask(image_path)
    img = mask.resize((256, 256))
    # Create the mask folder if it doesn't exist
    mask_directory = os.path.join(app.config['UPLOAD_FOLDER'], mask_folder)
    if not os.path.exists(mask_directory):
        os.makedirs(mask_directory)

    # Save the mask image with the desired filename in the 'mask' subdirectory
    mask_filename = os.path.join(mask_directory, os.path.basename(image_path).replace('_normalized.png', '_nuclick.png'))

    # Save the mask directly without converting to a PIL Image again
    img.save(mask_filename)

    return mask_filename


def latest_processed_SAM():
    processed_images = []

    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.endswith("_normalized.png"):
                processed_images.append(filename)

    if processed_images:
        # Sort the processed images by modification time 
        processed_images.sort(key=lambda x: os.path.getmtime(os.path.join(app.config['UPLOAD_FOLDER'], x)))

        # Get the name of the latest processed image
        latest_processed_name = os.path.splitext(processed_images[-1])[0]

        # Look for a segmented image with the same name in the 'SAM' subdirectory
        sam_subdirectory = os.path.join(app.config['UPLOAD_FOLDER'], 'SAM')
        segmented_image_path = os.path.join(sam_subdirectory, f"{latest_processed_name}_segmented.png")
        print(segmented_image_path)
        # Check if the segmented image exists
        if os.path.exists(segmented_image_path):
            print("Latest Processed Image:", segmented_image_path)
            return segmented_image_path
        else:
            print(f"No segmented image found for {latest_processed_name}")
    else:
        print("No processed images available")
        return None
    
def latest_processed_path():
    processed_images = []

    nuclick_directory = os.path.join(app.config['UPLOAD_FOLDER'], 'nuclick')

    if os.path.exists(nuclick_directory):
        for filename in os.listdir(nuclick_directory):
            if filename.endswith("_normalized.png"):
                processed_images.append(filename)

    if processed_images:
        # Sort the processed images by modification time (assuming newer ones are at the end)
        processed_images.sort(key=lambda x: os.path.getmtime(os.path.join(nuclick_directory, x)))

        # Return the relative path to the latest processed image in the "nuclick" subdirectory
        latest_processed_image = os.path.join('nuclick', processed_images[-1])
        print("Latest Processed Image Path:", latest_processed_image)
        return latest_processed_image
    else:
        print("No processed images available in the 'nuclick' subdirectory")
        return None
    
def nuclick_result_path():
    processed_images = []

    nuclick_directory = os.path.join(app.config['UPLOAD_FOLDER'], 'nuclick')

    if os.path.exists(nuclick_directory):
        for filename in os.listdir(nuclick_directory):
            if filename.endswith("_overlay.png"):
                processed_images.append(filename)

    if processed_images:
        # Sort the processed images by modification time (assuming newer ones are at the end)
        processed_images.sort(key=lambda x: os.path.getmtime(os.path.join(nuclick_directory, x)))

        # Return the desired path in the format uploads/nuclick/<filename>.png
        latest_processed_image = os.path.join('uploads', 'nuclick', processed_images[-1])
        print("Latest Processed Image Path:", latest_processed_image)
        return latest_processed_image
    else:
        print("No processed images available in the 'nuclick' subdirectory")
        return None
    
def color_normalization(image):
    # Convert image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Split the LAB image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # Merge the CLAHE enhanced L channel with the original A and B channels
    normalized_lab = cv2.merge((cl, a_channel, b_channel))

    # Convert the normalized LAB image back to RGB
    normalized_rgb = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2RGB)

    return normalized_rgb

def denoise_and_normalize(image_path, alpha=1.0, beta=0.15, contrast_alpha=1.5, contrast_beta=-40, target_resolution=(800, 600)):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to the target resolution
    image = cv2.resize(image, target_resolution)

    # Denoise and enhance contrast
    enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=contrast_alpha, beta=contrast_beta)

    # Apply color normalization
    normalized_image = color_normalization(enhanced_image)

    # Save the resized and preprocessed image
    normalized_filename = os.path.splitext(image_path)[0] + '_normalized.png'
    cv2.imwrite(normalized_filename, cv2.cvtColor(normalized_image, cv2.COLOR_RGB2BGR))

    return normalized_filename


def readImageAndGetClicks(image_file, cx, cy):
    """
    Read image from a file and get clicks.
    :param image_file: path to the image file
    :param cx: array of x-coordinates
    :param cy: array of y-coordinates
    :return: img - image array, cx - array of x-coordinates, cy - array of y-coordinates, imgPath - path to the image
    """
    imgPath = os.path.join(app.config['UPLOAD_FOLDER'], image_file)
    image = cv2.imread(imgPath)
    clone = image.copy()
    img = clone[:, :, ::-1]
    return img, cx, cy, imgPath




@app.route('/')
def index():
    return render_template('index.html')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif','tiff','svs'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#############################
# TISSUE TRAINING
from subprocess import Popen, PIPE
@app.route('/upload_train_tissue', methods=['POST'])
def upload_train_tissue():
    if 'files[]' not in request.files:
        return 'No file part'

    files = request.files.getlist('files[]')
    uploaded_files = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            uploaded_files.append(filename)
            print("Uploaded file:", filename)  # Print the filename

    print("Uploaded files:", uploaded_files)  # Print the list of uploaded filenames
    return jsonify({'uploaded_files': uploaded_files}), 200

# TRAIN TISSUE
@app.route('/upload_with_class', methods=['POST'])
def upload_file_with_class():
    if 'images' not in request.files:
        return jsonify({"error": "No file part"}), 400

    images = request.files.getlist('images')  # Get the list of images
    class_names = request.form.getlist('class_names')  # Get the list of class names
    dataset_name = request.form.get('dataset_name')  # Get the dataset name from the form

    if not dataset_name:
        return jsonify({"error": "Dataset name not provided"}), 400

    # Make sure the upload folder exists
    upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], dataset_name)
    upload_folder2=os.path.join(app.config['UPLOAD_FOLDER'])
    classnames_folder = os.path.join(upload_folder2, 'classnames')
    if not os.path.exists(classnames_folder):
        os.makedirs(classnames_folder)

    # Maintain order and uniqueness of class names
    unique_class_names = []
    seen = set()
    for class_name in class_names:
        if class_name not in seen:
            unique_class_names.append(class_name)
            seen.add(class_name)

    # Write unique class names to a text file while maintaining order
    classnames_file_path = os.path.join(classnames_folder, f"{dataset_name}.txt")
    with open(classnames_file_path, 'w') as f:
        f.write('\n'.join(unique_class_names))

    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    for i, image in enumerate(images):
        class_name = class_names[i]
        if not class_name:
            return jsonify({"error": "Class name not provided"}), 400

        class_directory = os.path.join(upload_folder, class_name)
        if not os.path.exists(class_directory):
            os.makedirs(class_directory)

        # Secure the filename
        filename = secure_filename(image.filename)
        # Prepend class name to filename
        filename_with_class = f"{class_name}_{filename}"
        # Construct the full file path
        file_path = os.path.join(class_directory, filename_with_class)
        # Save the file to the class directory
        image.save(file_path)

    return jsonify({"message": "Images uploaded and processed successfully"}), 200
#####tissue train
import subprocess
from subprocess import Popen, PIPE, STDOUT
import subprocess
from subprocess import Popen, PIPE, STDOUT
@app.route('/train_tissue_cnn', methods=['POST'])
def train_tissue_cnn():
    try:
        print("train_tissue_cnn endpoint hit.")  # Debug message
        
        # Get the absolute path to the train_tissue_cnn.py script
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_tissue_cnn.py')
        print("Script path:", script_path)  # Debug message
        
        # Execute train_tissue_cnn.py as a subprocess
        process = Popen(['python', script_path], stdout=PIPE, stderr=STDOUT, universal_newlines=True)
        
        # Read and print stdout and stderr in real-time
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                output_lines.append(output.strip())
        
        # Wait for the subprocess to finish
        process.wait()
        
        # Check if the process ran successfully
        if process.returncode == 0:
            # Extract the accuracy and loss from the output
            accuracy = None
            loss = None
            for line in output_lines:
                if "Training accuracy:" in line:
                    accuracy = float(line.split(":")[1].strip())
                if "Training loss:" in line:
                    loss = float(line.split(":")[1].strip())
            return jsonify({'message': 'train_tissue_cnn.py executed successfully', 'accuracy': accuracy, 'loss': loss}), 200
        else:
            return jsonify({'error': f'train_tissue_cnn.py failed with error: {process.returncode}'}), 500
    except Exception as e:
        return jsonify({'error': f'Error running train_tissue_cnn.py: {str(e)}'}), 500


#####train tissue backend
@app.route('/train_tissue_resnet', methods=['POST'])
def train_tissue_resnet():
    try:
        print("train_tissue_resnet endpoint hit.")  # Debug message
        
        # Get the absolute path to the train_tissue_cnn.py script
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resnet_train.py')
        print("Script path:", script_path)  # Debug message
        
        # Execute train_tissue_cnn.py as a subprocess
        process = Popen(['python', script_path], stdout=PIPE, stderr=STDOUT, universal_newlines=True)
        # Read and print stdout and stderr in real-time
        stdout_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                stdout_lines.append(output.strip())
                print(output.strip())
        
        # Wait for the subprocess to finish
        process.wait()
        
        # Check if the process ran successfully
        if process.returncode == 0:
            print("deleting outputs directory")
            shutil.rmtree('output')
            
            # Extract accuracy and loss from the stdout_lines
            accuracy = None
            loss = None
            for line in stdout_lines:
                if 'accuracy:' in line and 'val_accuracy:' in line:
                    # Parse the line to get accuracy and loss
                    parts = line.split('-')
                    for part in parts:
                        if 'accuracy' in part:
                            accuracy = float(part.split(':')[-1].strip())
                        if 'loss' in part:
                            loss = float(part.split(':')[-1].strip())
            
            if accuracy is not None and loss is not None:
                return jsonify({'message': 'resnet_train.py executed successfully', 'accuracy': accuracy, 'loss': loss}), 200
            else:
                return jsonify({'message': 'resnet_train.py executed successfully, but could not parse accuracy and loss'}), 200
        else:
            return jsonify({'error': f'resnet_train.py failed with error: {process.returncode}'}), 500
    except Exception as e:
        return jsonify({'error': f'Error running resnet_train.py: {str(e)}'}), 500

#####convnext integration#####
def extract_class_name(file_name):
    return file_name.split('_')[0]    

def flush_stdout():
    sys.stdout.flush()
from datasets import load_dataset

from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
)

from transformers import AutoImageProcessor
import torch.nn as nn
####train tissue backend##############
@app.route('/train_tissue_convNext', methods=['POST'])
def train_tissue_conVnext():
        # Specify the uploads directory
    script_dir = os.path.dirname(os.path.abspath(_file_))
    print(script_dir)
    uploads_folder = os.path.join(script_dir, 'uploads')
    print(uploads_folder)
    
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
    dataset_path = f"uploads/{dataset_name}"
# Count the number of subdirectories (classes) in the dataset
    num_of_classes = len([name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))])

    print("Preprocessing completed.","\n")
    flush_stdout()
    print('Proceeding to Train/Test Split', '\n')
    flush_stdout()
    # Remove existing train and test directories if they exist
    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


    # Lists to store file paths for train and test sets
    train_files = []
    test_files = []

    # Traverse all subdirectories within the obtained directory
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".jpg") or file.endswith(".png")or file.endswith(".tif"):  # Considering only image files
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

    print("dataset path:", dataset_path)

    print("Train/Test split completed successfully.")
    flush_stdout()
    
    dataset = load_dataset("imagefolder", data_dir=dataset_path)
    print(dataset["train"].features)

    # Visualize the first training example
    example = dataset["train"][0]
    example_image = example["image"]
    example_label = example["label"]

    print("Label:", example_label)
    example_image.show()

    # Get the number of rows (images) in train and test splits
    num_train_images = dataset["train"].num_rows
    num_test_images = dataset["test"].num_rows

    print("Number of images in train split:", num_train_images)
    print("Number of images in test split:", num_test_images)
    
    image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-tiny-224")
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    transform = Compose(
    [
    RandomResizedCrop(image_processor.size["shortest_edge"]),
    RandomHorizontalFlip(),
    ToTensor(),
    normalize
    ]
    )
    def train_transforms(examples):
        examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]

        return examples
    processed_dataset = dataset.with_transform(train_transforms)
    labels = dataset["train"].features["label"].names
    id2label = {k:v for k,v in enumerate(labels)}
    label2id = {v:k for k,v in enumerate(labels)}
    from torch.utils.data import DataLoader

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])

        return {"pixel_values": pixel_values, "labels": labels}

    dataloader = DataLoader(processed_dataset["train"], collate_fn=collate_fn, batch_size=16, shuffle=True)
    import torch

    batch = next(iter(dataloader))
    for k,v in batch.items():
        print(k,v.shape)
    
    from transformers import AutoModelForImageClassification

    model = AutoModelForImageClassification.from_pretrained("facebook/convnext-tiny-224",
                                                            id2label=id2label,
                                                            label2id=label2id,
                                                            ignore_mismatched_sizes=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_of_classes)
    from tqdm.notebook import tqdm

    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # move model to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.train()
    for epoch in range(1):
        print("Epoch:", epoch)
        correct = 0
        total = 0
        for idx, batch in enumerate(tqdm(dataloader)):
            # move batch to GPU
            batch = {k:v.to(device) for k,v in batch.items()}

            optimizer.zero_grad()

            # forward pass
            outputs = model(pixel_values=batch["pixel_values"],
                            labels=batch["labels"])

            loss, logits = outputs.loss, outputs.logits
            loss.backward()
            optimizer.step()

            # metrics
            total += batch["labels"].shape[0]
            predicted = logits.argmax(-1)
            correct += (predicted == batch["labels"]).sum().item()

            accuracy = correct/total

            if idx % 100 == 0:
                print(f"Loss after {idx} steps:", loss.item())
                print(f"Accuracy after {idx} steps:", accuracy)

        model_path = os.path.join("trained_models_tissue", f"{dataset_name}_ConVNeXt.pth")
        import torch
        torch.save(model.state_dict(), model_path)
        
    return jsonify({"message": "Model training initiated successfully"}), 200
        

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return "No file part", 400

    file = request.files['image']

    if file.filename == '':
        return "No selected file", 400

    if file and allowed_file(file.filename):
        # Make sure the upload folder exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        # Secure the filename and save the file to the server
        filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filename)

        # Check file extension
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension == '.tif' or file_extension == '.svs':
            ###displaying WSI image
            # Open the WSIReader and generate the thumbnail
            reader = WSIReader.open(filename)
            thumbnail = reader.slide_thumbnail(resolution=1.25, units="power")

            # Display the thumbnail using Matplotlib
            plt.imshow(thumbnail)
            plt.axis("off")

            # Save the thumbnail as a PNG file with the original filename in the 'uploads' directory
            output_filepath = os.path.join(f"{filename}.png")
            plt.savefig(output_filepath, bbox_inches='tight', pad_inches=0, transparent=True)

            # Close the Matplotlib plot
            plt.close()
            ###extract patches
            patch_size = (800, 800)
            stride = (800, 800)

            # Create a subdirectory for patches based on the original image name
            patches_dir = os.path.join('uploads', 'patches', os.path.basename(filename))
            os.makedirs(patches_dir, exist_ok=True)

            patch_extractor = patchextraction.get_patch_extractor(
                input_img=filename,
                method_name="slidingwindow",
                patch_size=patch_size,
                stride=stride,
            )

            for i, patch in enumerate(patch_extractor):
                # Generate a filename for each patch
                patch_filename = f"patch_{i + 1}.png"
                patch_filepath = os.path.join(patches_dir, patch_filename)

                # Save the patch as an image
                cv2.imwrite(patch_filepath, patch)
        else:
            # Perform denoising and normalization on the uploaded image with resizing
            normalized_image_path = denoise_and_normalize(filename)
            cache.set('latest_processed_image', normalized_image_path)
        return "File uploaded and processed successfully", 200
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'svs', 'tif', 'vms', 'vmu', 'ndpi', 'scn', 'mrxs', 'tiff', 'vsi', 'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_image(image_path, size):
    with Image.open(image_path) as img:
        resized_img = img.resize(size, Image.ANTIALIAS)
        resized_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'resized_' + os.path.basename(image_path))
        resized_img.save(resized_image_path)
        return resized_image_path

# @app.route('/upload_wsi', methods=['POST'])
# def upload_wsi():
#     if 'wsi' not in request.files:
#         return "No file part", 400

#     file = request.files['wsi']
#     if file.filename == '':
#         return "No selected file", 400

#     if file and allowed_file(file.filename):
#         if not os.path.exists(app.config['UPLOAD_FOLDER']):
#             os.makedirs(app.config['UPLOAD_FOLDER'])

#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)

#         # Generate and save thumbnail using WSIReader
#         reader = WSIReader.open(file_path)
#         thumbnail = reader.slide_thumbnail(resolution=1.25, units="power")
#         plt.imshow(thumbnail)
#         plt.axis("off")
#         thumbnail_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}_thumbnail.png")
#         plt.savefig(thumbnail_filepath, bbox_inches='tight', pad_inches=0)
#         plt.close()

#         # Create and download patches, integrate the create_patches logic here if needed
#         patches_dir, zip_filename = create_patches(file_path)
#         if patches_dir is None:
#             return "Error processing image", 500
#         return send_file(zip_filename, as_attachment=True)

#     return "Invalid file format", 400

# @app.route('/create_patches', methods=['POST'])
# def create_and_download_patches():
#     if 'wsi' not in request.files:
#         return "No file part", 400

#     file = request.files['wsi']
#     if file.filename == '':
#         return "No selected file", 400

#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)

#         patches_dir, zip_filename = create_patches(file_path)
#         if patches_dir is None:
#             return "Error processing image", 500
#         return send_file(zip_filename, as_attachment=True)

#     return "Invalid file format", 400

# def create_patches(filename):
    
#     patch_size = (800, 800)
#     stride = (800, 800)
#     patches_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'patches', os.path.splitext(os.path.basename(filename))[0])
#     os.makedirs(patches_dir, exist_ok=True)

#     reader = WSIReader.open(filename)
        
#     patch_extractor = patchextraction.get_patch_extractor(
#                 input_img=filename,
#                 method_name="slidingwindow",
#                 patch_size=patch_size,
#                 stride=stride,
#             )

#     for i, patch in enumerate(patch_extractor):
#         patch_filename = f"patch_{i + 1}.png"
#         patch_filepath = os.path.join(patches_dir, patch_filename)
#         cv2.imwrite(patch_filepath, patch)

#     # Zip the patches
#     zip_filename = os.path.join(app.config['UPLOAD_FOLDER'], os.path.splitext(os.path.basename(filename))[0] + '_patches.zip')
#     with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
#         for root, dirs, files in os.walk(patches_dir):
#             for file in files:
#                 file_path = os.path.join(root, file)
#                 zipf.write(file_path, arcname=os.path.relpath(file_path, os.path.join(patches_dir, '..')))

#     return patches_dir, zip_filename

###################################
@app.route('/upload_train_data', methods=['POST'])
def upload_train_data():
    train_folder = 'uploads/train'

    # Ensure the directory exists
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    files = request.files.getlist('files')
    labels = request.form.getlist('labels')

    csv_filename = os.path.join(train_folder, 'data.csv')
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        for file, label in zip(files, labels):
            # Validate the label to be either '0' or '1'
            if file.filename == '' or label not in ['0', '1']:
                continue

            filename = secure_filename(file.filename)
            file_path = os.path.join(train_folder, filename)
            file.save(file_path)
            writer.writerow([filename, label, file_path])

    return "Files and labels saved successfully", 200
############################################
@app.route('/display_mask')
def display_latest_processed():
    # Call the latest_processed function to get the path to the latest processed image
    image_path = create_mask()

    if image_path:
        # Use Flask's send_file to display the image in the browser
        return send_file(image_path, as_attachment=True)
    else:
        # If there are no processed images, render an error page or return an appropriate response
        return render_template('error.html', message='No processed images available')
    
@app.route('/display_vahadane')
def display_latest_vahadane():
    # Call the latest_processed function to get the path to the latest processed image
    image_path = create_visual_vahadane()

    if image_path:
        # Use Flask's send_file to display the image in the browser
        return send_file(image_path, as_attachment=True)
    else:
        # If there are no processed images, render an error page or return an appropriate response
        return render_template('error.html', message='No processed images available')
    
@app.route('/display_reinhard')
def display_latest_reinhard():
    # Call the latest_processed function to get the path to the latest processed image
    image_path = create_visual_reinhard()

    if image_path:
        # Use Flask's send_file to display the image in the browser
        return send_file(image_path, as_attachment=True)
    else:
        # If there are no processed images, render an error page or return an appropriate response
        return render_template('error.html', message='No processed images available')

@app.route('/display_nuclick_mask')
def display_latest_nuclick():
    # Call the latest_processed function to get the path to the latest processed image
    image_path = create_mask_nuclick()

    if image_path:
        # Use Flask's send_file to display the image in the browser
        return send_file(image_path, as_attachment=True)
    else:
        # If there are no processed images, render an error page or return an appropriate response
        return render_template('error.html', message='No processed images available')  


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/uploads/<path:filename>')
def uploads_file(filename):
    # Split the path into components
    path_components = filename.split('/')

    # Check if the path has 'SAM' as a component
    if 'SAM' in path_components:
        # Serve the file from the 'SAM' subdirectory
        return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], 'SAM'), path_components[-1])
    else:
        # Serve the file from the main 'uploads' directory
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/get_latest_processed_image')
def get_latest_processed_image():
    latest_processed_image = latest_processed()

    if latest_processed_image:
        print("Latest Processed Image:", latest_processed_image)
        return send_file(latest_processed_image, as_attachment=True)
    else:
        print("No processed image available")
        return jsonify(error="No processed image available"), 404
 
@app.route('/nuclick_result')
def get_nuclick_result_image():
    latest_processed_image = nuclick_result_path()

    if latest_processed_image:
        print("Latest Processed Image:", latest_processed_image)
        return send_file(latest_processed_image, as_attachment=True)
    else:
        print("No processed image available")
        return jsonify(error="No processed image available"), 404

@app.route('/get_latest_nuclick')
def get_latest_nuclick():
    latest_processed_image = latest_processed()

    if latest_processed_image:
        print("Latest Processed Image:", latest_processed_image)

        # Open the image using PIL
        img = Image.open(latest_processed_image)

        # Resize the image to 256x256
        img = img.resize((256, 256))

        # Save the resized image to a temporary file
        resized_image_path = os.path.join("uploads", "nuclick", os.path.basename(latest_processed_image))
        os.makedirs(os.path.dirname(resized_image_path), exist_ok=True)
        img.save(resized_image_path)

        # Send the resized image file
        return send_file(resized_image_path, as_attachment=True, download_name='resized_image.png')
    else:
        print("No processed image available")
        return jsonify(error="No processed image available"), 404

@app.route('/get_latest_SAM')
def get_SAM_Latest():
    latest_processed_image = latest_processed_SAM()

    if latest_processed_image:
        print("Latest Processed Image:", latest_processed_image)
        # Send the resized image file
        return send_file(latest_processed_image, as_attachment=True, download_name='resized_image.png')
    else:
        print("No processed image available")
        return jsonify(error="No processed image available"), 404  

@app.route('/get_latest_processed_image_path')
def get_latest_processed_image_path():
    latest_processed_image = latest_processed_path()

    if latest_processed_image:
        print("Latest Processed Image Path:", latest_processed_image)
        return latest_processed_image
    else:
        print("No processed image available")
        return "No processed image available", 404

@app.route('/processed_images', methods=['GET'])
def get_processed_images():
    processed_image_list = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.endswith("_normalized.png"):
                processed_image_list.append(url_for('uploaded_file', filename=filename))
    return jsonify(processed_image_list)


@app.route('/segmentation', methods=['POST'])
def nuclick():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    model_type = config.network
    weights_path = config.weights_path[0]
    print(weights_path)

    # loading models
    if (model_type.lower() == 'nuclick'):
        net = NuClick_NN(n_channels=5, n_classes=1)
    else:
        raise ValueError('Invalid model type. Acceptable networks are UNet or NuClick')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {model_type}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(weights_path, map_location=device))

    logging.info('Model loaded!')
    try:
        data = request.get_json()

        coordinates = data

        # Reading a specific image file
        image_file = latest_processed_path()

        # Coordinates from the frontend
        cx = [coord.get('x') for coord in coordinates]
        cy = [coord.get('y') for coord in coordinates]

        img, cx, cy, imgPath = readImageAndGetClicks(image_file, cx, cy)
        logging.info(cx)
        logging.info(cy)
        m, n = img.shape[0:2]
        img = np.asarray(img)[:, :, :3]
        img = np.moveaxis(img, 2, 0)
        clickMap, boundingBoxes = get_clickmap_boundingbox(cx, cy, m, n)
        patchs, nucPoints, otherPoints = get_patches_and_signals(img, clickMap, boundingBoxes, cx, cy, m, n)
        patchs = patchs / 255

        input = np.concatenate((patchs, nucPoints, otherPoints), axis=1, dtype=np.float32)
        input = torch.from_numpy(input)
        input = input.to(device=device, dtype=torch.float32)

        # Prediction with test time augmentation
        with torch.no_grad():
            output = net(input)
            output = torch.sigmoid(output)
            output = torch.squeeze(output, 1)
            preds = output.cpu().numpy()
        logging.info("Original images prediction, DONE!")

        masks = post_processing(preds, thresh=config.threshold, minSize=10, minHole=30, doReconstruction=True, nucPoints=nucPoints)

        # Generate instanceMap
        instanceMap = gen_instance_map(masks, boundingBoxes, m, n)
        img = np.moveaxis(img, 0, 2)

        # Convert instanceMap to uint8 using img_as_ubyte
        instanceMap_RGB = label2rgb(instanceMap, image=np.asarray(img)[:, :, :3], alpha=0.75, bg_label=0, bg_color=(0, 0, 0), image_alpha=1, kind='overlay')
        instanceMap_RGB = img_as_ubyte(instanceMap_RGB)  # Convert to uint8

        imsave(imgPath[:-4] + '_overlay.png', instanceMap_RGB)
        imsave(imgPath[:-4] + '_instances.png', instanceMap)
        # Save the segmentation results and annotations to MongoDB
        if image_file:
            nuclick_annotations_collection = mongo.db.Nuclick_Annotations
            nuclick_annotation = {
                'ImageID': os.path.basename(image_file),
                'imagePath': image_file,
                'cx': cx,
                'cy': cy
            }
            nuclick_annotations_collection.insert_one(nuclick_annotation)
            print("NuClick annotation saved to MongoDB")
        

        return "Segmentation completed successfully", 200
    except Exception as e:
        print("Error processing segmentation:", e)
        return "Error processing segmentation", 500

@app.route('/get_latest_nuclick')
def get_latest_nuclick_img():
    latest_processed_image = latest_processed_path()

    if latest_processed_image:
        print("Latest Processed Image:", latest_processed_image)
        return send_file(latest_processed_image, as_attachment=True)
    else:
        print("No processed image available")
        return jsonify(error="No processed image available"), 404

@app.route('/SAM', methods=['POST'])
def segment_image():
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"
    sam = sam_model_registry[MODEL_TYPE](checkpoint="weights/sam_vit_h_4b8939.pth").to(device=DEVICE)
    sam_model=SamPredictor(sam)
    logging.info("Model Loaded")

    data = request.get_json()
    print("Received data:", data)
    bounding_boxes = data.get('boundingBoxes', [])

    IMAGE_PATH = latest_processed()

    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mask_predictor = sam_model
    mask_predictor.set_image(image_rgb)
    # Use the original image as the result_image
    result_image = image_bgr.copy()

    # Initialize an image for segmentation annotations
    segmentation_image = result_image.copy()

    for box in bounding_boxes:
        box_coordinates = np.array([
            box['x'],
            box['y'],
            box['x'] + box['width'],
            box['y'] + box['height']
        ])

        masks, scores, logits = mask_predictor.predict(
            box=box_coordinates,
            multimask_output=True
        )

        mask_annotator = sv.MaskAnnotator(color=sv.Color.red(), color_lookup=sv.ColorLookup.INDEX)

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks
        )
        detections = detections[detections.area == np.max(detections.area)]

        # Apply segmentation annotations to the segmentation_image
        segmentation_image = mask_annotator.annotate(scene=segmentation_image, detections=detections)

    # Extract the file name and extension from the original file path
    original_filename, original_extension = os.path.splitext(os.path.basename(IMAGE_PATH))
    original_subdirectory = os.path.dirname(IMAGE_PATH)

    # Define the name for the segmented file
    result_image_filename = f"{original_filename}_segmented.png"

    # Save the final result image in the "SAM" subdirectory
    sam_subdirectory = "SAM"
    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], sam_subdirectory, result_image_filename)

    # Ensure the subdirectory exists, create it if not
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], sam_subdirectory, original_subdirectory), exist_ok=True)

    cv2.imwrite(result_image_path, segmentation_image)
    # Placeholder response
    result = {'message': 'Bounding boxes processed successfully'}


    # Save the SAM annotations to MongoDB
    if IMAGE_PATH:
        sam_annotations_collection = mongo.db.SAM_Annotations
        sam_annotation = {
            'ImageID': os.path.basename(IMAGE_PATH),
            'imagePath': result_image_path,
            'boundingBoxes': [{'x': box['x'], 'y': box['y'], 'width': box['width'], 'height': box['height']} for box in bounding_boxes]
        }
        sam_annotations_collection.insert_one(sam_annotation)
        print("SAM annotation saved to MongoDB")


    return jsonify(result)
#####################################
###Tissue Prediction
from transformers import AutoModelForImageClassification
#####################################
###Tissue Prediction
##New Function
#to fetch model files stored after training
@app.route('/modelsListTissue')
def get_models_list_tissue():
    trained_models_dir = "./trained_models_tissue"
    models_with_extension = [model for model in os.listdir(trained_models_dir) if model.endswith('.h5') or model.endswith('.pth')]
    models_without_extension = [os.path.splitext(model)[0] for model in models_with_extension]
    return jsonify({'models': models_without_extension})

from tensorflow.keras.preprocessing import image as KERAS_IMG
from PIL import Image as PIL_IMG
import io
import torch.nn as nn
from transformers import AutoModelForImageClassification
@app.route('/TissuePredict', methods=['POST'])
def tissue_predict_image():
    warnings.filterwarnings("ignore")

    # Get uploaded image and model selection from request
    uploaded_image = request.files['image']
    selected_model = request.form['model']

    if selected_model=="densenet121-kather100k" or selected_model=="resnet18-kather100k" or selected_model=="alexnet-kather100k":
        # Create the temporary directory if it doesn't exist
        temp_dir = Path("./tmp/")
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Create a unique identifier for this prediction
        prediction_id = str(uuid.uuid4())

        # Save the uploaded image to a temporary file
        temp_image_path = os.path.join(temp_dir, f"{prediction_id}_uploaded_image.tif")
        uploaded_image.save(temp_image_path)
        
        # Process the uploaded image using the selected model
        input_tile = np.array(Image.open(temp_image_path))

        # Obtain the mapping between the label ID and the class name
        label_dict = {
            "BACK": 0,
            "NORM": 1,
            "DEB": 2,
            "TUM": 3,
            "ADI": 4,
            "MUC": 5,
            "MUS": 6,
            "STR": 7,
            "LYM": 8,
        }
        class_names = list(label_dict.keys())
        class_labels = list(label_dict.values())

        # Process the image using the selected model (modify as needed)
        global_save_dir = Path("./tmp/")
        rmdir(global_save_dir / "tile_predictions")

        predictor = PatchPredictor(pretrained_model=selected_model, batch_size=32)
        tile_output = predictor.predict(
            imgs=[temp_image_path],
            mode="tile",
            merge_predictions=True,
            patch_input_shape=[224, 224],
            stride_shape=[224, 224],
            resolution=1,
            units="baseline",
            return_probabilities=True,
            save_dir=global_save_dir / "tile_predictions",
            on_gpu=False,  # Update this based on your configuration
        )

        # Extract information from output dictionary
        coordinates = tile_output[0]["coordinates"]
        predictions = tile_output[0]["predictions"]

        # Select 4 random indices (patches)
        rng = np.random.default_rng()  # Numpy Random Generator
        random_idx = rng.integers(0, len(predictions), (4,))

        patch_plot_files = []
        for i, idx in enumerate(random_idx):
            this_coord = coordinates[idx]
            this_prediction = predictions[idx]
            this_class = class_names[this_prediction]

            this_patch = input_tile[
                this_coord[1] : this_coord[3],
                this_coord[0] : this_coord[2],
            ]
            plt.subplot(2, 2, i + 1), plt.imshow(this_patch)
            plt.axis("off")
            plt.title(this_class)

            # Save the plot as a temporary file
            patch_plot_path = f"{prediction_id}patch_plot{i}.png"
            print(patch_plot_path)
            plt.savefig("patch_plots\\"+patch_plot_path, format='png')
            patch_plot_files.append(patch_plot_path)
            plt.close()

        # Visualization of merged image tile patch-level prediction
        tile_output[0]["resolution"] = 1.0
        tile_output[0]["units"] = "baseline"

        label_color_dict = {}
        label_color_dict[0] = ("empty", (0, 0, 0))
        colors = cm.get_cmap("Set1").colors
        for class_name, label in label_dict.items():
            label_color_dict[label + 1] = (class_name, 255 * np.array(colors[label]))
        pred_map = predictor.merge_predictions(
            temp_image_path,
            tile_output[0],
            resolution=1,
            units="baseline",
        )

        # Create overlay plot
        plt.imshow(input_tile)
        overlay = overlay_prediction_mask(
            input_tile,
            pred_map,
            alpha=0.5,
            label_info=label_color_dict,
            return_ax=True,
        )
        overlay_plot_path = f"{prediction_id}_overlay_plot.png"
        print("Overlay Plot File Path:", overlay_plot_path) 
        plt.savefig("overlay_plot\\"+overlay_plot_path, format='png')
        plt.close()

        # Send the patch plot files and overlay plot file to the client
        return jsonify({
            'prediction_id': prediction_id,
            'patch_plot_files': patch_plot_files,
            'overlay_plot': overlay_plot_path
        })
    else:
        if 'ResNet' in selected_model:
            prediction_id = str(uuid.uuid4())
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image.filename)
            uploaded_image.save(image_path)
            img = PIL_IMG.open(image_path)
            img = img.resize((224, 224))
            selected_model_path = f"./trained_models_tissue/{selected_model}.h5"
            loaded_model = load_model(selected_model_path)
            dataset_name = selected_model.rsplit('_', 1)[0]
            # Read target names from the corresponding .txt file
            target_names_file = os.path.join(app.config['UPLOAD_FOLDER'], 'classnames', f"{dataset_name}.txt")
            with open(target_names_file, 'r') as f:
                target_names = f.read().splitlines()

            # Convert the image to array and preprocess
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize the image

            # Make predictions using the loaded model
            predictions = loaded_model.predict(img_array)
            predicted_label = target_names[np.argmax(predictions)]

            # Display the image along with the predicted label
            plt.imshow(img)
            plt.title(f"Predicted Label: {predicted_label}")
            plt.axis('off')
            overlay_plot_path = f"{prediction_id}_overlay_plot.png"
            plt.savefig("overlay_plot\\" + overlay_plot_path, format='png')
            plt.close()
            # Send the patch plot files and overlay plot file to the client
            return jsonify({
                'prediction_id': prediction_id,
                'patch_plot_files': "NULL",
                'overlay_plot': overlay_plot_path
            })

        elif ('ConVnet' in selected_model) or ('ConVNet' in selected_model):
            # Load your image
            prediction_id = str(uuid.uuid4())
            img_stream = uploaded_image.stream
            img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)  
            imge = cv2.imdecode(img_array, cv2.IMREAD_COLOR) 
            print(selected_model)
            selected_model_path= f"./trained_models_tissue/{selected_model}.h5"
            print(selected_model_path)
            loaded_model = load_model(selected_model_path)
            # Preprocess the image 
            target_size = (224, 224) 
            imag = resize(imge, target_size, anti_aliasing=True)
            img_array = KERAS_IMG.img_to_array(imag)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.
            # Make predictions
            predictions = loaded_model.predict(img_array)

            # Convert predictions to class labels
            dataset_name = selected_model.rsplit('_', 1)[0]
            target_names_file = os.path.join(app.config['UPLOAD_FOLDER'], 'classnames', f"{dataset_name}.txt")
            with open(target_names_file, 'r') as f:
                target_names = f.read().splitlines()
            predicted_label = target_names[int(predictions[0][0] > 0.5)]

            # Display the image with the predicted label
            plt.imshow(imag)
            plt.axis('off')
            plt.title(f'Predicted Label: {predicted_label}')
            overlay_plot_path = f"{prediction_id}_overlay_plot.png"
            plt.savefig("overlay_plot\\"+overlay_plot_path, format='png')
            plt.close()
            # Send the patch plot files and overlay plot file to the client
            return jsonify({
                'prediction_id': prediction_id,
                'patch_plot_files': "NULL",
                'overlay_plot': overlay_plot_path
            })
        
        elif ('ConVNeXt' in selected_model):
            prediction_id = str(uuid.uuid4())
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image.filename)
            uploaded_image.save(image_path)
            image = Image.open(image_path).convert("RGB")

            selected_model_path = f"./trained_models_tissue/{selected_model}.pth"
            model = AutoModelForImageClassification.from_pretrained(
                "facebook/convnext-tiny-224",
                ignore_mismatched_sizes=True
            )
            dataset_name = selected_model.rsplit('_', 1)[0]
            target_names_file = os.path.join(app.config['UPLOAD_FOLDER'], 'classnames', f"{dataset_name}.txt")
            with open(target_names_file, 'r') as f:
                target_names = f.read().splitlines()

            num_classes = len(target_names)
            print(num_classes)
            original_num_features = model.classifier.in_features
            new_classifier = nn.Linear(original_num_features, num_classes)
            model.classifier = new_classifier


            print(selected_model_path)
            torch.save(model.state_dict(), selected_model_path)
            model.eval()
            

            # Load and preprocess the image
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            image_tensor = preprocess(image).unsqueeze(0)

            # Get predicted class index
            with torch.no_grad():
                output = model(image_tensor)
                predicted_class_index = torch.argmax(output.logits, dim=1).item()
                predicted_class_label = target_names[predicted_class_index]

            # Display the image along with the predicted label
            plt.imshow(image)
            plt.title(f"Predicted Label: {predicted_class_label}")
            plt.axis('off')
            overlay_plot_path = f"{prediction_id}_overlay_plot.png"
            plt.savefig("overlay_plot\\"+overlay_plot_path, format='png')
            plt.close()
            # Send the patch plot files and overlay plot file to the client
            return jsonify({
                'prediction_id': prediction_id,
                'patch_plot_files': "NULL",
                'overlay_plot': overlay_plot_path
            })

        #############################
@app.route('/download_patch_plot/<filename>')
def download_patch_plot(filename):
    patch_plot_path = "patch_plots\\"+filename
    print(patch_plot_path)
    return send_file(patch_plot_path, as_attachment=True)

@app.route('/download_overlay_plot/<filename>')
def download_overlay_plot(filename):
    overlay_plot_path = "overlay_plot\\"+filename
    return send_file(overlay_plot_path, as_attachment=True)

def rmdir(dir_path: str | Path) -> None:
    """Helper function to delete directory."""
    if Path(dir_path).is_dir():
        shutil.rmtree(dir_path)
        logger.info("Removing directory %s", dir_path)



# # Load the pretrained UNet model
# unet_model = models.segmentation.deeplabv3_resnet101(pretrained=True)
# unet_model.eval()

# # Preprocess input image for UNet
# unet_transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])



import math 


def count_cells(mask):
    # Convert the mask to binary image
    _, binary_mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours((binary_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for counting cells
    minimum_area = 200
    average_cell_area = 650
    connected_cell_area = 1000
    cells = 0

    # Loop through each contour
    for c in contours:
        area = cv2.contourArea(c)
        if area > minimum_area:
            if area > connected_cell_area:
                cells += math.ceil(area / average_cell_area)
            else:
                cells += 1

    return cells

import sys


####cell prediction####
#to fetch model files stored after training
@app.route('/modelsList')
def get_models_list():
    trained_models_dir = "./trained_models"
    
    # List all files and folders in the directory
    all_entries = os.listdir(trained_models_dir)
    
    # Separate files and folders
    models_with_extension = [model for model in all_entries if model.endswith('.h5')]
    folders = [entry for entry in all_entries if os.path.isdir(os.path.join(trained_models_dir, entry))]
    
    # Remove extensions from model files
    models_without_extension = [os.path.splitext(model)[0] for model in models_with_extension]
    
    return jsonify({'models': models_without_extension, 'folders': folders})

from stardist import random_label_cmap
import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from stardist.models import StarDist2D
from stardist.plot import render_label
from csbdeep.utils import normalize
from tqdm import tqdm
import random
import os
from io import BytesIO

@app.route('/CellPredict', methods=['POST'])
def cell_predict_image():
    warnings.filterwarnings("ignore")
    if not os.path.exists("./CellPredict"):
        os.makedirs("./CellPredict")


    save_path = "./CellPredict/result.png"
    uploaded_image = request.files['image']
    selected_model = request.form['model']

    print("Selected Model is:", selected_model)

    print("images loaded")
    if selected_model=="StarDist":
        # Save the uploaded image to the uploads directory
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image.filename)
        uploaded_image.save(image_path)
        
        # Load the uploaded image
        img = imread(image_path)
        # Load or define model
        demo_model = True
        if demo_model:
            print("NOTE: This is loading a previously trained demo model!\n"
                "      Please set the variable 'demo_model = False' to load your own trained model.",
                file=sys.stderr, flush=True)
            model = StarDist2D.from_pretrained('2D_demo')
        else:
            model = StarDist2D(None, name='stardist', basedir='models')

        # Make predictions on the image
        img_normalized = normalize(img, 1, 99.8)
        labels, details = model.predict_instances(img_normalized)

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot original image
        axes[0].imshow(img_normalized if img_normalized.ndim == 2 else img_normalized[..., 0], clim=(0, 1), cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Plot result
        axes[1].imshow(img_normalized if img_normalized.ndim == 2 else img_normalized[..., 0], clim=(0, 1), cmap='gray')
        axes[1].imshow(labels, cmap=random_label_cmap(), alpha=0.5)
        axes[1].set_title('Result')
        axes[1].axis('off')

        # Save the plot as result.png in the same directory
        plt.savefig(save_path)

    
    
    else:
        # Load the uploaded image
        img_stream = uploaded_image.stream
        img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)  
        

        # Determine if the selected model is an H5 file or a folder
        selected_model_path_h5 = f"./trained_models/{selected_model}.h5"
        selected_model_path_folder = f"./trained_models/{selected_model}"
        
        if os.path.exists(selected_model_path_h5):
            selected_model = load_model(selected_model_path_h5)
                    # Preprocess the image
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            target_size = (256, 256)
            image = resize(image, target_size, anti_aliasing=True)

            # Make predictions
            predicted_mask = selected_model.predict(np.expand_dims(image, axis=0))[0]

            # Display the results and save the image
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title('Input Image')
            plt.imshow(image)
            plt.subplot(1, 2, 2)
            plt.title('Detected Cells')
            plt.imshow(np.squeeze(predicted_mask), cmap='gray')
            if save_path:
                plt.savefig(save_path)
            
            # Count cells in the predicted mask
            num_cells = count_cells(predicted_mask)
            print("Number of cells:", num_cells)
                    # Preprocess the image
            target_size = (256, 256)
            image = resize(image, target_size, anti_aliasing=True)

            # Make predictions
            predicted_mask = selected_model.predict(np.expand_dims(image, axis=0))[0]

            # Display the results and save the image
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title('Input Image')
            plt.imshow(image)
            plt.subplot(1, 2, 2)
            plt.title('Detected Cells')
            plt.imshow(np.squeeze(predicted_mask), cmap='gray')
            if save_path:
                plt.savefig(save_path)
            
            # Count cells in the predicted mask
            num_cells = count_cells(predicted_mask)
            print("Number of cells:", num_cells)

        elif os.path.exists(selected_model_path_folder):
            np.random.seed(42)
            lbl_cmap = random_label_cmap()

            img = imread(BytesIO(img_array))


            # Normalize the image
            n_channel = 1 if img.ndim == 2 else img.shape[-1]
            axis_norm = (0, 1)  # normalize channels independently
            # axis_norm = (0,1,2) # normalize channels jointly
            if n_channel > 1:
                print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
            img = normalize(img, 1, 99.8, axis=axis_norm)

            # Load the pre-trained StarDist model
            model = StarDist2D(None, name=selected_model, basedir="./trained_models/")

            # Predict the instances in the image
            labels, details = model.predict_instances(img)

            # Visualize the results
            plt.figure(figsize=(8, 8))
            plt.imshow(img if img.ndim == 2 else img[..., 0], clim=(0, 1), cmap='gray')
            plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)
            plt.axis('off')
            if save_path:
                plt.savefig(save_path)
        else:
            raise ValueError(f"Model path {selected_model_path_h5} or {selected_model_path_folder} does not exist")



    # elif selected_model == 'Stardist2d':
    #     pass

    # else:
    #     return jsonify({'error': 'Invalid model selection'})

    # Return the filename of the saved segmented image
    result_image_url = f'CellPredict/result.png'
    # segmented_image_filename = os.path.basename(save_path)
    return jsonify({'num_cells': 67, 'segmented_image_filename': result_image_url})
@app.route('/CellPredict/<filename>')
def cell_predict_image_file(filename):
    return send_from_directory('./CellPredict', filename)


@app.route('/save-annotated-image', methods=['POST'])
def save_annotated_image():
    try:
        data = request.get_json()
        image_url = data.get('imageUrl')
        annotations = data.get('annotations', [])

        # Decode and save the image
        if image_url:
            encoded_image = image_url.split(',')[1]  # Extract the base64 string
            decoded_image = base64.b64decode(encoded_image)

            # Generate a unique filename using the first annotation's imageID
            if annotations:
                image_id = annotations[0]['imageID'].replace(":", "_").replace("T", "_").replace("Z", "")
                filename = f"{image_id}.png"
            else:
                filename = f"image_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"

            image_save_path = os.path.join('annotated_tissue_images', filename)
            with open(image_save_path, 'wb') as f:
                f.write(decoded_image)

            # Inserting imageID and annotated image into database, in table Annotated_Tissue_Images
            # Insert image data into the Annotated_Tissue_Images collection
            annotated_tissue_images_collection = mongo.db.Annotated_Tissue_Images
            annotated_tissue_images_collection.insert_one({'imageID': image_id, 'imagePath': image_save_path})

        


        # Process and save annotations data to MongoDB
        tissue_annotations_collection = mongo.db.Tissue_Annotations  # Accessing the collection
        for annot in annotations:
            # Insert each annotation into the MongoDB collection
            tissue_annotations_collection.insert_one(annot)
            print(f"Annotation inserted into MongoDB: {annot}")


        # Insert annotations data into MongoDB
        if annotations:
            db = mongo.db
            tissue_annotations = db["Tissue_Annotations"]  # Access the collection
            for annot in annotations:
                annot['imagePath'] = image_save_path  # Include the path of the saved image
                tissue_annotations.insert_one(annot)  # Insert the annotation into the collection


        return jsonify({'message': 'Image and annotations saved successfully'})
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/upload_with_class_cell', methods=['POST'])
def upload_file_with_class_cell():
    if 'images' not in request.files and 'masks' not in request.files:
        return jsonify({"error": "No file part"}), 400

    images = request.files.getlist('images')  # Get the list of images
    masks = request.files.getlist('masks')  # Get the list of masks
    class_names = request.form.getlist('class_names')  # Get the list of class names
    print(class_names)
    dataset_name = request.form.get('dataset_name')  # Get the dataset name from the form

    if not dataset_name:
        return jsonify({"error": "Dataset name not provided"}), 400

    # Make sure the upload folder exists
    upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], dataset_name)
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    for i, image in enumerate(images):
        class_name = class_names[i]
        if not class_name:
            return jsonify({"error": "Class name not provided"}), 400

        class_directory = os.path.join(upload_folder, class_name)
        if not os.path.exists(class_directory):
            os.makedirs(class_directory)

        # Secure the filename and save the file to the class directory
        filename = os.path.join(class_directory, secure_filename(image.filename))
        image.save(filename)

    for i, mask in enumerate(masks):
        class_name = class_names[i]
        if not class_name:
            return jsonify({"error": "Class name not provided"}), 400

        mask_directory = os.path.join(upload_folder, "masks")
        if not os.path.exists(mask_directory):
            os.makedirs(mask_directory)

        # Secure the filename and save the file to the class directory
        filename = os.path.join(mask_directory, secure_filename(mask.filename))
        mask.save(filename)

    return jsonify({"message": "Images and masks uploaded and processed successfully"}), 200



#####model training for Cell Detection#############
def unet(input_shape):
    inputs = tf.keras.layers.Input(input_shape)
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    # Contraction path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Expansive path
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

@app.route('/check_training_status')
def check_training_status():
    # This should return the current status of the training process
    # Example:
    return jsonify({"status": "in_progress", "message": "Training is still in progress"})

@app.route('/train_cell', methods=['POST'])
def train_model_cell():
    global training_status
    training_status = {'status': 'in_progress', 'message': 'Training has started.'}

    # Retrieve hyperparameters from the request
    Epochs = int(request.form['epochs'])
    Learning_rate = float(request.form['learning_rate'])
    Batch_size = int(request.form['batch_size'])
    selected_model = request.form['modelName']

    print("selected model is:", selected_model)
    # Specify the uploads directory
    uploads_folder = UPLOAD_FOLDER

    # Check if the uploads directory exists
    if not os.path.exists(uploads_folder):
        print("Error: 'uploads' directory not found")
        # Handle the error gracefully, e.g., by exiting the script or raising an exception
    else:
        print("Uploads folder found successfully.")

    # Get the most recently created directory in the uploads folder
    try:
        # Filter out only directories
        directories = [f for f in os.listdir(uploads_folder) if os.path.isdir(os.path.join(uploads_folder, f))]
        
        # Find the most recently created directory
        most_recent_dir = max(directories, key=lambda f: os.path.getctime(os.path.join(uploads_folder, f)))
        dataset_name = most_recent_dir
        print(f"Dataset '{dataset_name}' ")
    except ValueError:
        print("Error: No directories found in 'uploads' folder")
        # Handle the error gracefully, e.g., by exiting the script or raising an exception
    except Exception as e:
        print(f"Error: {e}")
        

    # Define image and mask directories within the dataset folder
    dataset_folder = os.path.join(uploads_folder, dataset_name)
    image_subdir = os.path.join(dataset_folder, 'images')
    mask_subdir = os.path.join(dataset_folder, 'masks')
    print(image_subdir)
    print(mask_subdir)

    if (selected_model=="unet"):
        from pathlib import Path
        ###imports for cell model training
        
        from pathlib import Path
        from skimage.io import imread
        from skimage.transform import resize
        import numpy as np
        import random
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        # Load image and mask files
        X = sorted(glob.glob(image_subdir + '/*.*'))
        Y = sorted(glob.glob(mask_subdir + '/*.*'))

        # Extract file extensions
        extensions_X = [Path(x).suffix.lower() for x in X]
        extensions_Y = [Path(y).suffix.lower() for y in Y]

        # Check image types and use appropriate reading functions
        if all(ext in ['.tif', '.tiff'] for ext in extensions_X + extensions_Y):
            imread_func = lambda x: imread(x, plugin='tifffile')
        else:
            imread_func = imread

        # Extract image IDs from filenames
        image_ids_X = [Path(x).stem for x in X]
        image_ids_Y = [Path(y).stem for y in Y]

        # Find common image IDs
        common_ids = set(image_ids_X).intersection(set(image_ids_Y))

        # Filter X and Y to only keep corresponding images and masks
        X = [x for x, image_id_x in zip(X, image_ids_X) if image_id_x in common_ids]
        Y = [y for y, image_id_y in zip(Y, image_ids_Y) if image_id_y in common_ids]

        # Check if number of images matches number of masks
        assert len(X) == len(Y)
        # Load and preprocess the training and validation data
        target_size = (256, 256)  # Define the target size for resizing
        X_train_data = [resize(imread_func(x), target_size, anti_aliasing=True) for x in X]
        Y_train_data = [resize(imread_func(y), target_size, anti_aliasing=False) for y in Y]

        # Apply correction to ensure single-channel masks
        Y_train_data_binary = []
        for mask in Y_train_data:
            # Convert to grayscale if necessary
            if mask.shape[-1] > 1:
                mask = np.mean(mask, axis=-1, keepdims=True)
            # Apply binary thresholding
            binary_mask = (mask > 0).astype(np.float32)
            Y_train_data_binary.append(binary_mask)

        # Convert lists to NumPy arrays
        X_train_data = np.array(X_train_data)
        Y_train_data = np.array(Y_train_data_binary)
        print("Shape of Y_train_data:", Y_train_data.shape)

        # Define input shape
        input_shape = X_train_data[0].shape

        # Build and compile the model
        model = unet(input_shape)
        model.summary()

        #X_train, X_test, Y_train, Y_test = train_test_split(X_resized, Y_resized, test_size=0.1, random_state=42)
        X_train, X_test, Y_train, Y_test = train_test_split(X_train_data , Y_train_data , test_size=0.1, random_state=42)
        ################################ Define ModelCheckpoint callback
        trained_models_path = os.path.join(app.root_path, 'trained_models')
        model_checkpoint_path = os.path.join(trained_models_path, f'{dataset_name}_model.h5')
        checkpointer = tf.keras.callbacks.ModelCheckpoint( model_checkpoint_path, 
                                                        verbose=1, 
                                                        save_best_only=True)

        # Define other callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
            tf.keras.callbacks.TensorBoard(log_dir='logs')
        ]

        # Combine all callbacks
        all_callbacks = callbacks + [checkpointer]

    
        # Train the model with callbacks
        history = model.fit(X_train, Y_train, 
                            validation_split=0.1, 
                            batch_size=Batch_size, 
                            epochs=Epochs, 
                            callbacks=all_callbacks)
        ####################################
        training_status = {'status': 'completed', 'message': 'Training completed successfully.'}
        history_dict = history.history
        training_metrics = {
            'loss': history_dict['loss'][-1],
            'accuracy': history_dict['accuracy'][-1],
            # Add more metrics as needed
        }
    


        return jsonify({"message": "Model training initiated successfully","history": history_dict, "training_metrics": training_metrics}), 200
    
    elif (selected_model=="stardist"):
        ###added import to first line of file
        import sys
        import numpy as np
        import matplotlib
        matplotlib.rcParams["image.interpolation"] = 'none'
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        from tifffile import imread
        from csbdeep.utils import Path, normalize
        from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
        from stardist.matching import matching, matching_dataset
        from stardist.models import Config2D, StarDist2D, StarDistData2D

        # Random color map labels
        np.random.seed(42)
        lbl_cmap = random_label_cmap()

        # Read input image and corresponding mask names
        X = sorted(glob.glob(image_subdir +'/*.tif'))
        Y = sorted(glob.glob(mask_subdir + '/*.tif'))

        # Read images and masks using their names.
        X = list(map(imread, X))
        Y = list(map(imread, Y))

        n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

        # Normalize input images and fill holes in masks
        axis_norm = (0, 1)
        if n_channel > 1:
            print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
            sys.stdout.flush()

        X = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X)]
        Y = [fill_label_holes(y) for y in tqdm(Y)]

        # Split to train, val, and test
        assert len(X) > 2, "not enough training data"
        rng = np.random.RandomState(42)
        ind = rng.permutation(len(X))
        n_test = max(1, int(round(0.15 * len(ind))))
        n_val = max(1, int(round(0.15 * (len(ind) - n_test))))
        ind_test = ind[:n_test]
        ind_val = ind[n_test:n_test + n_val]
        ind_train = ind[n_test + n_val:]
        X_test, Y_test = [X[i] for i in ind_test], [Y[i] for i in ind_test]
        X_val, Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]
        X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
        print('number of images: %3d' % len(X))
        print('- training:       %3d' % len(X_trn))
        print('- validation:     %3d' % len(X_val))
        print('- testing:        %3d' % len(X_test))

        # Define the config by setting some parameter values
        n_rays = 32  # Number of radial directions for the star-convex polygon.

        # Use OpenCL-based computations for data generator during training (requires 'gputools')
        use_gpu = False and gputools_available()

        # Predict on subsampled grid for increased efficiency and larger field of view
        grid = (2, 2)

        conf = Config2D(
            n_rays=n_rays,
            grid=grid,
            use_gpu=use_gpu,
            n_channel_in=n_channel,
        )
        print(conf)
        vars(conf)

        if use_gpu:
            from csbdeep.utils.tf import limit_gpu_memory
            limit_gpu_memory(0.8)

        trained_models_path = os.path.join(app.root_path, 'trained_models')
        model = StarDist2D(conf, name=dataset_name, basedir=trained_models_path)
        print(f"Base directory for storing model weights: {model.basedir}")

        median_size = calculate_extents(list(Y), np.median)
        fov = np.array(model._axes_tile_overlap('YX'))
        print(f"median object size:      {median_size}")
        print(f"network field of view :  {fov}")
        if any(median_size > fov):
            print("WARNING: median object size larger than field of view of the neural network.")

        # Define a few augmentation methods
        def random_fliprot(img, mask):
            assert img.ndim >= mask.ndim
            axes = tuple(range(mask.ndim))
            perm = tuple(np.random.permutation(axes))
            img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
            mask = mask.transpose(perm)
            for ax in axes:
                if np.random.rand() > 0.5:
                    img = np.flip(img, axis=ax)
                    mask = np.flip(mask, axis=ax)
            return img, mask

        def random_intensity_change(img):
            img = img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
            return img

        def augmenter(x, y):
            x, y = random_fliprot(x, y)
            x = random_intensity_change(x)
            sig = 0.02 * np.random.uniform(0, 1)
            x = x + sig * np.random.normal(0, 1, x.shape)
            return x, y

        # Capture training history
        history = model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augmenter, epochs=Epochs, steps_per_epoch=100)

        model.optimize_thresholds(X_val, Y_val)
        training_status = {'status': 'completed', 'message': 'Training completed successfully.'}

        # Extract history dictionary and training metrics
        history_dict = history.history
        history_dict = {key: [float(value) for value in values] for key, values in history_dict.items()}
        training_metrics = {
            'loss': history_dict['loss'][-1],
            'val_loss': history_dict['val_loss'][-1],
            'accuracy': history_dict.get('accuracy', [None])[-1],
            'val_accuracy': history_dict.get('val_accuracy', [None])[-1],
        }

        # Pass the history and training metrics to the frontend
        return jsonify({
            "message": "Model training initiated successfully",
            "history": history_dict,
            "training_metrics": training_metrics
        }), 200







##########PROJECT CREATION CODE###############
app.config["MONGO_URI"] = "mongodb://localhost:27017/pathosyncDB" 
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

mongo = PyMongo(app)

# Checking if the MongoDB connection was successful
if mongo.db is None:
    raise Exception("MongoDB connection failed. Please check the MongoDB URI and server status.")

projects_collection = mongo.db.projects
images_collection = mongo.db.images

@app.route('/projects/create', methods=['POST'])
def create_project():
    try:
        data = request.json
        print("Received data:", data)  # Log request data
        if not data:
            raise ValueError("No data provided")

        new_project = {
            'name': data['name'],
            'numClasses': data['numClasses'],
            'datasetType': data['datasetType'],
            'classNames': data['classNames']
        }
        result = projects_collection.insert_one(new_project)  # Add the new project to MongoDB
        new_project['_id'] = str(result.inserted_id)

        return jsonify({'message': 'Project created successfully', 'project': new_project}), 201
    except ValueError as ve:
        print("ValueError:", str(ve))  # Log ValueError
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print("Error:", str(e))  # Log any other errors
        return jsonify({'error': 'An error occurred while creating the project'}), 500

@app.route('/projects', methods=['GET'])
def get_projects():
    projects = list(projects_collection.find())
    for project in projects:
        project['_id'] = str(project['_id'])
        project['images'] = list(images_collection.find({'project_id': project['_id']}))
        for image in project['images']:
            image['_id'] = str(image['_id'])
    return jsonify(projects)

@app.route('/projects/<project_id>/upload', methods=['POST'])
def upload_image(project_id):
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        files = request.files.getlist('file')
        saved_files = []
        
        for file in files:
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            image_data = {
                '_id': str(ObjectId()),
                'filename': filename,
                'filepath': file_path
            }
            projects_collection.update_one(
                {'_id': ObjectId(project_id)},
                {'$push': {'images': image_data}}
            )
            saved_files.append(image_data)

        return jsonify({'message': 'Files uploaded successfully', 'images': saved_files}), 201
    except Exception as e:
        return jsonify({'error': 'An error occurred while uploading the file(s)'}), 500  


# ########cloud storage new function########
# # Configuration
# from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
# from datetime import datetime, timedelta

# AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=pathosynccloud;AccountKey=d11O+yRWcggC7ZRpw152aVBRnA8szlAp3pIwafKI2ozqJkmIxOdbCnoHNbdoFAw4uQwLMKQa9HzO+AStyRb9lw==;EndpointSuffix=core.windows.net"
# AZURE_STORAGE_CONTAINER_NAME = "wsi-images"

# blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

# @app.route('/get_sas_token', methods=['POST'])
# def get_sas_token():
#     data = request.get_json()
#     filename = data['filename']
#     filetype = data['filetype']
    
#     sas_token = generate_blob_sas(
#         account_name=blob_service_client.account_name,
#         container_name=AZURE_STORAGE_CONTAINER_NAME,
#         blob_name=filename,
#         account_key=blob_service_client.credential.account_key,
#         permission=BlobSasPermissions(write=True),
#         expiry=datetime.utcnow() + timedelta(hours=1)
#     )
    
#     blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{AZURE_STORAGE_CONTAINER_NAME}/{filename}"
    
#     return jsonify({'url': blob_url, 'token': sas_token})

# def download_blob_to_local(filename):
#     blob_client = blob_service_client.get_blob_client(container=AZURE_STORAGE_CONTAINER_NAME, blob=filename)
#     local_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     print(local_path)
#     with open(local_path, "wb") as download_file:
#         download_file.write(blob_client.download_blob().readall())
#     return local_path

# def create_patches(local_file_path):
#     patch_size = (800, 800)
#     stride = (800, 800)
#     patches_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'project_dataset', os.path.splitext(os.path.basename(local_file_path))[0])
#     os.makedirs(patches_dir, exist_ok=True)

#     reader = WSIReader.open(local_file_path)
        
#     patch_extractor = patchextraction.get_patch_extractor(
#                 input_img=local_file_path,
#                 method_name="slidingwindow",
#                 patch_size=patch_size,
#                 stride=stride,
#             )

#     for i, patch in enumerate(patch_extractor):
#         patch_filename = f"patch_{i + 1}.png"
#         print(patch_filename)
#         patch_filepath = os.path.join(patches_dir, patch_filename)
#         cv2.imwrite(patch_filepath, patch)

#     return patches_dir

# @app.route('/create_patches', methods=['POST'])
# def create_and_save_patches():
#     data = request.get_json()
#     filename = data['filename']
    
#     if not filename:
#         return "No filename provided", 400

#     try:
#         local_file_path = download_blob_to_local(filename)
#         patches_dir = create_patches(local_file_path)
#         if patches_dir is None:
#             return "Error processing image", 500
#         return {"patches_directory": patches_dir}, 200
#     except Exception as e:
#         return str(e), 500


if __name__ == '__main__':
    app.run(debug=True)
    

=======
import cv2
import numpy as np
from skimage.io import imsave
import matplotlib.pyplot as plt
from models import UNet, NuClick_NN
import torch
import base64
import logging
from skimage.color import label2rgb
from skimage import img_as_ubyte
from PIL import Image
from datetime import datetime
import os
from config import DemoConfig as config
from utils.process import post_processing, gen_instance_map
os.add_dll_directory(r"C:\Users\bisma\Downloads\NuClick\nuclick-flask-app\app\dll_files")
from tiatoolbox.tools.tissuemask import MorphologicalMasker
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.stainnorm import VahadaneNormalizer,ReinhardNormalizer
from tiatoolbox import data
from tiatoolbox.tools import patchextraction
from utils.misc import get_clickmap_boundingbox
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import supervision as sv
from utils.guiding_signals import get_patches_and_signals
import torch
from flask import Flask, request, render_template, send_from_directory
from flask_cors import CORS
from flask import jsonify
from skimage import io
from skimage import color
import skimage.io
from skimage import io, exposure, img_as_ubyte
from werkzeug.utils import secure_filename
from flask import url_for, send_file
from flask_caching import Cache
import tiatoolbox.utils.visualization as vis_utils

from flask_pymongo import PyMongo
from flask import Flask
from pymongo import MongoClient
from bson import ObjectId

app = Flask(__name__)
CORS(app)
cache = Cache(app)

#Database Configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/PathoSync"
mongo = PyMongo(app)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def latest_processed():
    processed_images = []

    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.endswith("_normalized.png"):
                processed_images.append(filename)

    if processed_images:
        # Sort the processed images by modification time (assuming newer ones are at the end)
        processed_images.sort(key=lambda x: os.path.getmtime(os.path.join(app.config['UPLOAD_FOLDER'], x)))

        # Return the path to the latest processed image
        latest_processed_image = os.path.join(app.config['UPLOAD_FOLDER'], processed_images[-1])
        print("Latest Processed Image:", latest_processed_image)
        return latest_processed_image
    else:
        print("No processed images available")
        return None

# Functions to make image masks
def visualize_mask(image):

    pil_image = Image.open(image)
    image_np = np.array(pil_image)

    # Apply morphological dilation using OpenCV
    kernel = np.ones((5, 5), np.uint8)  # Define a kernel for dilation
    dilated_image = cv2.dilate(image_np, kernel)

    # Perform tissue masking using TIAToolbox
    masker = MorphologicalMasker()
    mask = masker.fit_transform([image_np])[0]

    # Convert the tissue mask to a PIL image
    mask_pil_image = Image.fromarray(mask)


    return mask_pil_image

def create_mask():
    image_path = latest_processed()
    mask_folder = 'mask'

    # Create a mask using visualize_mask function
    mask = visualize_mask(image_path)

    # Create the mask folder if it doesn't exist
    mask_directory = os.path.join(app.config['UPLOAD_FOLDER'], mask_folder)
    if not os.path.exists(mask_directory):
        os.makedirs(mask_directory)

    # Save the mask image with the desired filename in the 'mask' subdirectory
    mask_filename = os.path.join(mask_directory, os.path.basename(image_path).replace('_normalized.png', '_masked.png'))

    # Save the mask directly without converting to a PIL Image again
    mask.save(mask_filename)

    return mask_filename


# Functions for vahadane normalization VahadaneNormalizer
def vahadane_mask(image):
    vahad_vis= VahadaneNormalizer()
    pil_image = Image.open(image)
    img_array = np.array(pil_image)
    target_image = data.stain_norm_target()
    vahad_vis.fit(target_image)
    norm_img=vahad_vis.transform(img_array)
    return norm_img

def create_visual_vahadane():
    image_path = latest_processed()
    mask_folder = 'vahadane'

    # Create a mask using visualize_mask function
    visual = vahadane_mask(image_path)

    # Create the mask folder if it doesn't exist
    mask_directory = os.path.join(app.config['UPLOAD_FOLDER'], mask_folder)
    if not os.path.exists(mask_directory):
        os.makedirs(mask_directory)

    # Save the mask image with the desired filename in the 'mask' subdirectory
    mask_filename = os.path.join(mask_directory, os.path.basename(image_path).replace('_normalized.png', '_vahadane.png'))

    # Save the mask directly without converting to a PIL Image again
    visual_image = Image.fromarray((visual * 255).astype(np.uint8))
    visual_image.save(mask_filename)

    return mask_filename

# Functions for reinhard normalization ReinhardNormalizer
def reinhard_mask(image):
    vahad_vis= ReinhardNormalizer()
    pil_image = Image.open(image)
    img_array = np.array(pil_image)
    target_image=Image.open("target_image.png")
    vahad_vis.fit(target_image)
    norm_img=vahad_vis.transform(img_array)
    return norm_img

def create_visual_reinhard():
    image_path = latest_processed()
    mask_folder = 'reinhard'

    # Create a mask using visualize_mask function
    visual = reinhard_mask(image_path)

    # Create the mask folder if it doesn't exist
    mask_directory = os.path.join(app.config['UPLOAD_FOLDER'], mask_folder)
    if not os.path.exists(mask_directory):
        os.makedirs(mask_directory)

    # Save the mask image with the desired filename in the 'mask' subdirectory
    mask_filename = os.path.join(mask_directory, os.path.basename(image_path).replace('_normalized.png', '_reinhard.png'))

    # Save the mask directly without converting to a PIL Image again
    visual_image = Image.fromarray((visual * 255).astype(np.uint8))
    visual_image.save(mask_filename)

    return mask_filename

def create_mask_nuclick():
    image_path = latest_processed()
    mask_folder = 'mask'

    # Create a mask using visualize_mask function
    mask = visualize_mask(image_path)
    img = mask.resize((256, 256))
    # Create the mask folder if it doesn't exist
    mask_directory = os.path.join(app.config['UPLOAD_FOLDER'], mask_folder)
    if not os.path.exists(mask_directory):
        os.makedirs(mask_directory)

    # Save the mask image with the desired filename in the 'mask' subdirectory
    mask_filename = os.path.join(mask_directory, os.path.basename(image_path).replace('_normalized.png', '_nuclick.png'))

    # Save the mask directly without converting to a PIL Image again
    img.save(mask_filename)

    return mask_filename


def latest_processed_SAM():
    processed_images = []

    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.endswith("_normalized.png"):
                processed_images.append(filename)

    if processed_images:
        # Sort the processed images by modification time (assuming newer ones are at the end)
        processed_images.sort(key=lambda x: os.path.getmtime(os.path.join(app.config['UPLOAD_FOLDER'], x)))

        # Get the name of the latest processed image
        latest_processed_name = os.path.splitext(processed_images[-1])[0]

        # Look for a segmented image with the same name in the 'SAM' subdirectory
        sam_subdirectory = os.path.join(app.config['UPLOAD_FOLDER'], 'SAM')
        segmented_image_path = os.path.join(sam_subdirectory, f"{latest_processed_name}_segmented.png")
        print(segmented_image_path)
        # Check if the segmented image exists
        if os.path.exists(segmented_image_path):
            print("Latest Processed Image:", segmented_image_path)
            return segmented_image_path
        else:
            print(f"No segmented image found for {latest_processed_name}")
    else:
        print("No processed images available")
        return None
    
def latest_processed_path():
    processed_images = []

    nuclick_directory = os.path.join(app.config['UPLOAD_FOLDER'], 'nuclick')

    if os.path.exists(nuclick_directory):
        for filename in os.listdir(nuclick_directory):
            if filename.endswith("_normalized.png"):
                processed_images.append(filename)

    if processed_images:
        # Sort the processed images by modification time (assuming newer ones are at the end)
        processed_images.sort(key=lambda x: os.path.getmtime(os.path.join(nuclick_directory, x)))

        # Return the relative path to the latest processed image in the "nuclick" subdirectory
        latest_processed_image = os.path.join('nuclick', processed_images[-1])
        print("Latest Processed Image Path:", latest_processed_image)
        return latest_processed_image
    else:
        print("No processed images available in the 'nuclick' subdirectory")
        return None
    
def nuclick_result_path():
    processed_images = []

    nuclick_directory = os.path.join(app.config['UPLOAD_FOLDER'], 'nuclick')

    if os.path.exists(nuclick_directory):
        for filename in os.listdir(nuclick_directory):
            if filename.endswith("_overlay.png"):
                processed_images.append(filename)

    if processed_images:
        # Sort the processed images by modification time (assuming newer ones are at the end)
        processed_images.sort(key=lambda x: os.path.getmtime(os.path.join(nuclick_directory, x)))

        # Return the desired path in the format uploads/nuclick/<filename>.png
        latest_processed_image = os.path.join('uploads', 'nuclick', processed_images[-1])
        print("Latest Processed Image Path:", latest_processed_image)
        return latest_processed_image
    else:
        print("No processed images available in the 'nuclick' subdirectory")
        return None
    
def color_normalization(image):
    # Convert image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Split the LAB image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # Merge the CLAHE enhanced L channel with the original A and B channels
    normalized_lab = cv2.merge((cl, a_channel, b_channel))

    # Convert the normalized LAB image back to RGB
    normalized_rgb = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2RGB)

    return normalized_rgb

def denoise_and_normalize(image_path, alpha=1.0, beta=0.15, contrast_alpha=1.5, contrast_beta=-40, target_resolution=(800, 600)):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to the target resolution
    image = cv2.resize(image, target_resolution)

    # Denoise and enhance contrast
    enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=contrast_alpha, beta=contrast_beta)

    # Apply color normalization
    normalized_image = color_normalization(enhanced_image)

    # Save the resized and preprocessed image
    normalized_filename = os.path.splitext(image_path)[0] + '_normalized.png'
    cv2.imwrite(normalized_filename, cv2.cvtColor(normalized_image, cv2.COLOR_RGB2BGR))

    return normalized_filename
    
def readImageAndGetClicks(image_file, cx, cy):
    """
    Read image from a file and get clicks.
    :param image_file: path to the image file
    :param cx: array of x-coordinates
    :param cy: array of y-coordinates
    :return: img - image array, cx - array of x-coordinates, cy - array of y-coordinates, imgPath - path to the image
    """
    imgPath = os.path.join(app.config['UPLOAD_FOLDER'], image_file)
    image = cv2.imread(imgPath)
    clone = image.copy()
    img = clone[:, :, ::-1]
    return img, cx, cy, imgPath




@app.route('/')
def index():
    return render_template('index.html')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif','tiff','svs'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return "No file part", 400

    file = request.files['image']

    if file.filename == '':
        return "No selected file", 400

    if file and allowed_file(file.filename):
        # Make sure the upload folder exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        # Secure the filename and save the file to the server
        filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filename)

        # Check file extension
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension == '.tif' or file_extension == '.svs':
            ###displaying WSI image
            # Open the WSIReader and generate the thumbnail
            reader = WSIReader.open(filename)
            thumbnail = reader.slide_thumbnail(resolution=1.25, units="power")

            # Display the thumbnail using Matplotlib
            plt.imshow(thumbnail)
            plt.axis("off")

            # Save the thumbnail as a PNG file with the original filename in the 'uploads' directory
            output_filepath = os.path.join(f"{filename}.png")
            plt.savefig(output_filepath, bbox_inches='tight', pad_inches=0, transparent=True)

            # Close the Matplotlib plot
            plt.close()
            ###extract patches
            patch_size = (800, 800)
            stride = (800, 800)

            # Create a subdirectory for patches based on the original image name
            patches_dir = os.path.join('uploads', 'patches', os.path.basename(filename))
            os.makedirs(patches_dir, exist_ok=True)

            patch_extractor = patchextraction.get_patch_extractor(
                input_img=filename,
                method_name="slidingwindow",
                patch_size=patch_size,
                stride=stride,
            )

            for i, patch in enumerate(patch_extractor):
                # Generate a filename for each patch
                patch_filename = f"patch_{i + 1}.png"
                patch_filepath = os.path.join(patches_dir, patch_filename)

                # Save the patch as an image
                cv2.imwrite(patch_filepath, patch)
        else:
            # Perform denoising and normalization on the uploaded image with resizing
            normalized_image_path = denoise_and_normalize(filename)
            cache.set('latest_processed_image', normalized_image_path)
        return "File uploaded and processed successfully", 200

@app.route('/display_mask')
def display_latest_processed():
    # Call the latest_processed function to get the path to the latest processed image
    image_path = create_mask()

    if image_path:
        # Use Flask's send_file to display the image in the browser
        return send_file(image_path, as_attachment=True)
    else:
        # If there are no processed images, render an error page or return an appropriate response
        return render_template('error.html', message='No processed images available')
    
@app.route('/display_vahadane')
def display_latest_vahadane():
    # Call the latest_processed function to get the path to the latest processed image
    image_path = create_visual_vahadane()

    if image_path:
        # Use Flask's send_file to display the image in the browser
        return send_file(image_path, as_attachment=True)
    else:
        # If there are no processed images, render an error page or return an appropriate response
        return render_template('error.html', message='No processed images available')
    
@app.route('/display_reinhard')
def display_latest_reinhard():
    # Call the latest_processed function to get the path to the latest processed image
    image_path = create_visual_reinhard()

    if image_path:
        # Use Flask's send_file to display the image in the browser
        return send_file(image_path, as_attachment=True)
    else:
        # If there are no processed images, render an error page or return an appropriate response
        return render_template('error.html', message='No processed images available')

@app.route('/display_nuclick_mask')
def display_latest_nuclick():
    # Call the latest_processed function to get the path to the latest processed image
    image_path = create_mask_nuclick()

    if image_path:
        # Use Flask's send_file to display the image in the browser
        return send_file(image_path, as_attachment=True)
    else:
        # If there are no processed images, render an error page or return an appropriate response
        return render_template('error.html', message='No processed images available')  


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/uploads/<path:filename>')
def uploads_file(filename):
    # Split the path into components
    path_components = filename.split('/')

    # Check if the path has 'SAM' as a component
    if 'SAM' in path_components:
        # Serve the file from the 'SAM' subdirectory
        return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], 'SAM'), path_components[-1])
    else:
        # Serve the file from the main 'uploads' directory
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/get_latest_processed_image')
def get_latest_processed_image():
    latest_processed_image = latest_processed()

    if latest_processed_image:
        print("Latest Processed Image:", latest_processed_image)
        return send_file(latest_processed_image, as_attachment=True)
    else:
        print("No processed image available")
        return jsonify(error="No processed image available"), 404
 
@app.route('/nuclick_result')
def get_nuclick_result_image():
    latest_processed_image = nuclick_result_path()

    if latest_processed_image:
        print("Latest Processed Image:", latest_processed_image)
        return send_file(latest_processed_image, as_attachment=True)
    else:
        print("No processed image available")
        return jsonify(error="No processed image available"), 404

@app.route('/get_latest_nuclick')
def get_latest_nuclick():
    latest_processed_image = latest_processed()

    if latest_processed_image:
        print("Latest Processed Image:", latest_processed_image)

        # Open the image using PIL
        img = Image.open(latest_processed_image)

        # Resize the image to 256x256
        img = img.resize((256, 256))

        # Save the resized image to a temporary file
        resized_image_path = os.path.join("uploads", "nuclick", os.path.basename(latest_processed_image))
        os.makedirs(os.path.dirname(resized_image_path), exist_ok=True)
        img.save(resized_image_path)

        # Send the resized image file
        return send_file(resized_image_path, as_attachment=True, download_name='resized_image.png')
    else:
        print("No processed image available")
        return jsonify(error="No processed image available"), 404

@app.route('/get_latest_SAM')
def get_SAM_Latest():
    latest_processed_image = latest_processed_SAM()

    if latest_processed_image:
        print("Latest Processed Image:", latest_processed_image)
        # Send the resized image file
        return send_file(latest_processed_image, as_attachment=True, download_name='resized_image.png')
    else:
        print("No processed image available")
        return jsonify(error="No processed image available"), 404  

@app.route('/get_latest_processed_image_path')
def get_latest_processed_image_path():
    latest_processed_image = latest_processed_path()

    if latest_processed_image:
        print("Latest Processed Image Path:", latest_processed_image)
        return latest_processed_image
    else:
        print("No processed image available")
        return "No processed image available", 404

@app.route('/processed_images', methods=['GET'])
def get_processed_images():
    processed_image_list = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.endswith("_normalized.png"):
                processed_image_list.append(url_for('uploaded_file', filename=filename))
    return jsonify(processed_image_list)


@app.route('/segmentation', methods=['POST'])
def nuclick():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    model_type = config.network
    weights_path = config.weights_path[0]
    print(weights_path)

    # loading models
    if (model_type.lower() == 'nuclick'):
        net = NuClick_NN(n_channels=5, n_classes=1)
    elif (model_type.lower() == 'unet'):
        net = UNet(n_channels=5, n_classes=1)
    else:
        raise ValueError('Invalid model type. Acceptable networks are UNet or NuClick')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {model_type}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(weights_path, map_location=device))

    logging.info('Model loaded!')
    try:
        data = request.get_json()

        # Assuming data is a list of coordinates
        coordinates = data

        # Reading a specific image file
        image_file = latest_processed_path()

        # Coordinates from the frontend
        cx = [coord.get('x') for coord in coordinates]
        cy = [coord.get('y') for coord in coordinates]

        img, cx, cy, imgPath = readImageAndGetClicks(image_file, cx, cy)
        logging.info(cx)
        logging.info(cy)
        m, n = img.shape[0:2]
        img = np.asarray(img)[:, :, :3]
        img = np.moveaxis(img, 2, 0)
        clickMap, boundingBoxes = get_clickmap_boundingbox(cx, cy, m, n)
        patchs, nucPoints, otherPoints = get_patches_and_signals(img, clickMap, boundingBoxes, cx, cy, m, n)
        patchs = patchs / 255

        input = np.concatenate((patchs, nucPoints, otherPoints), axis=1, dtype=np.float32)
        input = torch.from_numpy(input)
        input = input.to(device=device, dtype=torch.float32)

        # Prediction with test time augmentation
        with torch.no_grad():
            output = net(input)
            output = torch.sigmoid(output)
            output = torch.squeeze(output, 1)
            preds = output.cpu().numpy()
        logging.info("Original images prediction, DONE!")

        masks = post_processing(preds, thresh=config.threshold, minSize=10, minHole=30, doReconstruction=True, nucPoints=nucPoints)

        # Generate instanceMap
        instanceMap = gen_instance_map(masks, boundingBoxes, m, n)
        img = np.moveaxis(img, 0, 2)

        # Convert instanceMap to uint8 using img_as_ubyte
        instanceMap_RGB = label2rgb(instanceMap, image=np.asarray(img)[:, :, :3], alpha=0.75, bg_label=0, bg_color=(0, 0, 0), image_alpha=1, kind='overlay')
        instanceMap_RGB = img_as_ubyte(instanceMap_RGB)  # Convert to uint8

        imsave(imgPath[:-4] + '_overlay.png', instanceMap_RGB)
        imsave(imgPath[:-4] + '_instances.png', instanceMap)
        # Save the segmentation results and annotations to MongoDB
        if image_file:
            nuclick_annotations_collection = mongo.db.Nuclick_Annotations
            nuclick_annotation = {
                'ImageID': os.path.basename(image_file),
                'imagePath': image_file,
                'cx': cx,
                'cy': cy
            }
            nuclick_annotations_collection.insert_one(nuclick_annotation)
            print("NuClick annotation saved to MongoDB")
        

        return "Segmentation completed successfully", 200
    except Exception as e:
        print("Error processing segmentation:", e)
        return "Error processing segmentation", 500

@app.route('/get_latest_nuclick')
def get_latest_nuclick_img():
    latest_processed_image = latest_processed_path()

    if latest_processed_image:
        print("Latest Processed Image:", latest_processed_image)
        return send_file(latest_processed_image, as_attachment=True)
    else:
        print("No processed image available")
        return jsonify(error="No processed image available"), 404

@app.route('/SAM', methods=['POST'])
def segment_image():
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"
    sam = sam_model_registry[MODEL_TYPE](checkpoint="weights/sam_vit_h_4b8939.pth").to(device=DEVICE)
    sam_model=SamPredictor(sam)
    logging.info("Model Loaded")

    data = request.get_json()
    print("Received data:", data)
    bounding_boxes = data.get('boundingBoxes', [])

    IMAGE_PATH = latest_processed()

    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mask_predictor = sam_model
    mask_predictor.set_image(image_rgb)
    # Use the original image as the result_image
    result_image = image_bgr.copy()

    # Initialize an image for segmentation annotations
    segmentation_image = result_image.copy()

    for box in bounding_boxes:
        box_coordinates = np.array([
            box['x'],
            box['y'],
            box['x'] + box['width'],
            box['y'] + box['height']
        ])

        masks, scores, logits = mask_predictor.predict(
            box=box_coordinates,
            multimask_output=True
        )

        mask_annotator = sv.MaskAnnotator(color=sv.Color.red(), color_lookup=sv.ColorLookup.INDEX)

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks
        )
        detections = detections[detections.area == np.max(detections.area)]

        # Apply segmentation annotations to the segmentation_image
        segmentation_image = mask_annotator.annotate(scene=segmentation_image, detections=detections)

    # Extract the file name and extension from the original file path
    original_filename, original_extension = os.path.splitext(os.path.basename(IMAGE_PATH))
    original_subdirectory = os.path.dirname(IMAGE_PATH)

    # Define the name for the segmented file
    result_image_filename = f"{original_filename}_segmented.png"

    # Save the final result image in the "SAM" subdirectory
    sam_subdirectory = "SAM"
    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], sam_subdirectory, result_image_filename)

    # Ensure the subdirectory exists, create it if not
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], sam_subdirectory, original_subdirectory), exist_ok=True)

    cv2.imwrite(result_image_path, segmentation_image)
    # Placeholder response
    result = {'message': 'Bounding boxes processed successfully'}


    # Save the SAM annotations to MongoDB
    if IMAGE_PATH:
        sam_annotations_collection = mongo.db.SAM_Annotations
        sam_annotation = {
            'ImageID': os.path.basename(IMAGE_PATH),
            'imagePath': result_image_path,
            'boundingBoxes': [{'x': box['x'], 'y': box['y'], 'width': box['width'], 'height': box['height']} for box in bounding_boxes]
        }
        sam_annotations_collection.insert_one(sam_annotation)
        print("SAM annotation saved to MongoDB")


    return jsonify(result)

@app.route('/save-annotated-image', methods=['POST'])
def save_annotated_image():
    try:
        data = request.get_json()
        image_url = data.get('imageUrl')
        annotations = data.get('annotations', [])

        # Decode and save the image
        if image_url:
            encoded_image = image_url.split(',')[1]  # Extract the base64 string
            decoded_image = base64.b64decode(encoded_image)

            # Generate a unique filename using the first annotation's imageID
            if annotations:
                image_id = annotations[0]['imageID'].replace(":", "_").replace("T", "_").replace("Z", "")
                filename = f"{image_id}.png"
            else:
                filename = f"image_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"

            image_save_path = os.path.join('annotated_tissue_images', filename)
            with open(image_save_path, 'wb') as f:
                f.write(decoded_image)

            # Inserting imageID and annotated image into database, in table Annotated_Tissue_Images
            # Insert image data into the Annotated_Tissue_Images collection
            annotated_tissue_images_collection = mongo.db.Annotated_Tissue_Images
            annotated_tissue_images_collection.insert_one({'imageID': image_id, 'imagePath': image_save_path})


        # Process and save annotations data to MongoDB
        tissue_annotations_collection = mongo.db.Tissue_Annotations  # Accessing the collection
        for annot in annotations:
            # Insert each annotation into the MongoDB collection
            tissue_annotations_collection.insert_one(annot)
            print(f"Annotation inserted into MongoDB: {annot}")
        # Insert annotations data into MongoDB
        if annotations:
            db = mongo.db
            tissue_annotations = db["Tissue_Annotations"]  # Access the collection
            for annot in annotations:
                annot['imagePath'] = image_save_path  # Include the path of the saved image
                tissue_annotations.insert_one(annot)  # Insert the annotation into the collection
        return jsonify({'message': 'Image and annotations saved successfully'})
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)
>>>>>>> 8655ceccc37e8fd8d0bdcbd17d190dc036418d41
