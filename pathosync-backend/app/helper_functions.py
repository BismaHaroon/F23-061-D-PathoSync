import numpy as np
import cv2
from PIL import Image
import os
os.add_dll_directory(r"C:\Users\Saamiya M\anaconda3\envs\tiatoolbox-dev\Lib\site-packages\openslide-win64-20231011\bin")
from tiatoolbox.tools.tissuemask import MorphologicalMasker
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.stainnorm import VahadaneNormalizer,ReinhardNormalizer
from tiatoolbox import data
from flask_caching import Cache




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

def denoise_and_normalize(image_path, alpha=1.0, beta=0.15, contrast_alpha=1.5, contrast_beta=-40, target_resolution=(600, 400)):
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


