import numpy as np
from skimage.io import imsave
import matplotlib.pyplot as plt
from models import UNet, NuClick_NN
import torch
import base64
import logging
from skimage.color import label2rgb
from skimage import img_as_ubyte 
import cv2
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
