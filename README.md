# PathoSync: Enhancing Pathology with Synchronized AI

![landingpage](https://github.com/BismaHaroon/F23-061-D-PathoSync/assets/125141049/f750419c-832c-49d7-9fc0-b5a9b04003b8)


PathoSync is a revolutionary project designed to transform the landscape of digital pathology by introducing a cutting-edge platform that seamlessly integrates artificial intelligence (AI) with the expertise of pathologists. This innovative solution aims to address the challenges inherent in conventional pathology tools, offering a user-friendly experience that elevates diagnostic accuracy and streamlines workflows for medical professionals.

## Key Features
### Intuitive Interface
PathoSync prioritizes accessibility, providing a user-friendly interface that empowers pathologists with varying technical expertise.

### Image Upload
- **Upload Interface:** PathoSync features a user-friendly image upload interface for seamless transfer of medical images.
- **File Handling:** Supports various image file types, including png, jpg, jpeg, and gif.
- **Standard Resolution:** Images are standardized to a resolution of 600x400 pixels.


https://github.com/BismaHaroon/F23-061-D-PathoSync/assets/125141049/d59a2ccc-6c72-4a1a-a454-dd525ecfab26



### Image Preprocessing
- **De-noising:** Includes de-noising algorithms to improve image clarity.
- **Contrast Enhancement:** Enhances image contrast for improved visualization.
- **Image Resizing:** Resizes images for standardized processing across the platform.
- **Color Normalization:** Employs color normalization techniques for consistent color representation.
  <img width="457" alt="image" src="https://github.com/BismaHaroon/F23-061-D-PathoSync/assets/125575282/11b62da9-1585-40dc-845b-73cf63d7a520">


### Cellular Annotation
- **NuClick:** Incorporates NuClick for point annotations and labeling at the cellular level.

https://github.com/BismaHaroon/F23-061-D-PathoSync/assets/125141049/3f74734d-f64f-49a6-955e-c483a5a74bd5


  <img width="672" alt="image" src="https://github.com/BismaHaroon/F23-061-D-PathoSync/assets/125575282/7717c11d-0069-4b47-8917-44500affa006">

- **SAM:** Utilizes SAM for bounding box annotations and labeling, enhancing cellular annotation accuracy.


https://github.com/BismaHaroon/F23-061-D-PathoSync/assets/125141049/76ed5b81-80b2-4df3-822c-7b0b1b7708eb


  <img width="491" alt="image" src="https://github.com/BismaHaroon/F23-061-D-PathoSync/assets/125575282/115b9a90-bd6a-4eb2-9594-7b93f6d142d9">
  



### Tissue Annotation


https://github.com/BismaHaroon/F23-061-D-PathoSync/assets/125141049/6510e1a6-06fe-4c6a-8fd5-6ff5ce7d3898



- Annotations in the form of rectangles, lines, and freehand drawings.
- Selectable, resizable, movable annotations for different tissue structures.
- Customizable Colors: Personalize annotation colors for better visualization.
- Labels and List of Labels: Allows labeling of tissue annotations with an associated list of labels.
  
  <img width="198" alt="image" src="https://github.com/BismaHaroon/F23-061-D-PathoSync/assets/125575282/9985ebca-2c3a-47ef-a7b7-1fb9e3f4983b">



### WSI Upload & Preprocessing
- **Image Upload:** Facilitates the upload of Whole Slide Images (WSI) for comprehensive pathology analysis.
  <img width="255" alt="image" src="https://github.com/BismaHaroon/F23-061-D-PathoSync/assets/125575282/e905af69-747c-469e-b0ca-43c2a811947e">

- **Preprocessing:** Streamlines WSI preprocessing by efficiently extracting patches for detailed examination.
  <img width="571" alt="image" src="https://github.com/BismaHaroon/F23-061-D-PathoSync/assets/125575282/717ee6c5-73bf-4286-b083-bed37f90febb">

### WSI Patch Annotations
- **Tissue Prediction**

https://github.com/BismaHaroon/F23-061-D-PathoSync/assets/125141049/3268db09-f364-4d75-863b-e64dcea162f0

- **WSI Annotation**



Uploading WSI Annotate - Made with Clipchamp.mp4…



### Visualization
- **Visual Magnification:** Offers a visual magnification feature for detailed views of cellular and tissue structures.
- **Cell Masking:** Provides tissue masking capabilities for enhanced visibility and focus during the annotation process.

### Customizable Cell Detection Pipeline
- **Upload Custom Datasets:** Easily upload custom datasets for tailored analysis and model training.


https://github.com/BismaHaroon/F23-061-D-PathoSync/assets/125141049/39965fd8-f7d9-496d-b78b-dd692e9388aa



- **Build Custom Models:** Construct custom models to suit specific pathology requirements.
- **Advanced Model Training Features:**  Utilize advanced model training features such as epochs, batch size, and learning rate for optimal results.


https://github.com/BismaHaroon/F23-061-D-PathoSync/assets/125141049/f6e95136-a888-425d-9639-9c46184e0ffc



### Cell Prediction
- **Predict using saved models:** Once trained, select models for cell prediction, upload test images, and view results instantly.


https://github.com/BismaHaroon/F23-061-D-PathoSync/assets/125141049/70d796fa-1fc5-4800-b068-7f0a0b64b5cd


## Customizable Tissue Classification Pipeline
-**Custom Classes Creation:** Users can create custom classes tailored to the unique characteristics of their datasets, ensuring precise classification.

-**Custom Dataset Integration:** PathoSync allows the seamless upload of images from custom datasets, facilitating personalized model training and analysis.

-**Custom Model Building:** Users have the flexibility to build custom models suited to their specific requirements, enabling advanced classification tasks.

-**Advanced Model Training Features:** PathoSync provides the capability to input advanced model training features such as epochs, batch size, and learning rate, enabling users to fine-tune their models for optimal performance and desired results.





## Tissue Classification
- Once the model is trained, it is saved and can be conveniently selected for prediction tasks.
- Users can upload test images, and the results are promptly displayed for analysis.
- Users have the option to choose from pre-trained DenseNet, ResNet, and AlexNet models trained on the Kather100k dataset, further expanding the versatility of the classification process.

https://github.com/BismaHaroon/F23-061-D-PathoSync/assets/125141049/d2db69fd-70c7-4e6c-98b5-1e1b5acfbab6


## Table of Contents

- [Getting Started](#getting-started)
- [Usage](#usage)
- [Built With](#built-with)


## Getting Started
To get started with PathoSync, follow the instructions below.

1. Clone the repository.
   ```bash
   git clone https://github.com/BismaHaroon/F23-061-D-PathoSync.git
   cd pathosync
   ```

2. Install dependencies.
   ```bash
   # Frontend
   cd Pathosync-frontend
   npm install

   # Backend
   cd Pathosync-backend
   pip install -r requirements.txt
   ```
3. Install SAM & Nuclick weights.
   ```bash
   SAM weights: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   Nuclick weights: https://drive.google.com/file/d/1JBK3vWsVC4DxbcStukwnKNZm-vCSLdOb/view
   Copy them to the 'weights' folder in the Pathosync-backend directory
   ```
3. Run the application.
   ```bash
   # Frontend
   cd Pathosync-frontend
   npm start

   # Backend
   cd Pathosync-backend
   python app.py
   ```

## Usage

1. **Image Upload:**
   - Access the user-friendly upload interface.
   - Select the medical image file (png, jpg, jpeg, gif).
   - Confirm the upload, and PathoSync will standardize the image to 800x600 pixels.
     
2. **Image Preprocessing:**
   - Benefit from de-noising algorithms for improved image clarity.
   - Enhance image contrast for better visualization.

3. **Annotation:**
   - Engage in cellular annotation using NuClick for precise point annotations.
   - Utilize SAM for bounding box annotations to enhance cellular annotation accuracy.
   - Perform tissue annotations with customizable rectangles, ellipses, lines, polygons, and freehand drawings.
   - Personalize colors and add labels to enhance annotation visibility and organization.

4. **WSI Analysis:**
   - Upload Whole Slide Images for comprehensive pathology analysis.
   - Streamline preprocessing by efficiently extracting tiles for detailed examination.

5. **Visualization:**
   - Zoom in for visual magnification to inspect cellular and tissue structures in detail.
   - Apply tissue masking for enhanced visibility during the annotation process.


   - Resize images and employ color normalization techniques for consistent color representation.

6. **React Frontend and Flask Backend:**
   - Run the frontend and backend separately for a modular and scalable architecture.
   - Follow provided instructions to set up and start each component.


## Built With

- [Python](https://www.python.org/) - The primary programming language used for backend development.
- [Flask](https://flask.palletsprojects.com/) - A micro web framework for building the backend of the application.
- [PyTorch](https://pytorch.org/) - A deep learning library utilized for implementing cutting-edge AI capabilities.
- [React](https://reactjs.org/) - A JavaScript library used for building the interactive and dynamic frontend.
- [scikit-image](https://scikit-image.org/) - An image processing library in Python for image preprocessing tasks.
- [TIAToolbox](https://github.com/TissueImageAnalytics/tiatoolbox) - A toolkit for computational pathology, used for stain normalization and color manipulation.
- [NumPy](https://numpy.org/) - A fundamental package for scientific computing with Python, used for array operations.
- [OpenCV](https://opencv.org/) - An open-source computer vision and machine learning software library, used for image processing tasks.
- [Node.js](https://nodejs.org/) - A JavaScript runtime for executing JavaScript code on the server side, used for managing frontend dependencies.
- [npm](https://www.npmjs.com/) - A package manager for JavaScript, used for installing and managing frontend libraries and packages.
- [Meta: Segment Anything Model](https://github.com/facebookresearch/segment-anything) - An advanced segmentation model for precise annotation and region identification.
- [NuClick](https://github.com/mostafajahanifar/nuclick_torch) - An advanced CNN-based approach to speed up collecting annotations for microscopic objects requiring minimum interaction from the annotator.

These technologies were carefully selected to create a robust, scalable, and efficient platform that combines the strengths of various tools and frameworks for a seamless user experience in digital pathology.





