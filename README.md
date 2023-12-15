# PathoSync: Enhancing Pathology with Synchronized AI

PathoSync is a revolutionary project designed to transform the landscape of digital pathology by introducing a cutting-edge platform that seamlessly integrates artificial intelligence (AI) with the expertise of pathologists. This innovative solution aims to address the challenges inherent in conventional pathology tools, offering a user-friendly experience that elevates diagnostic accuracy and streamlines workflows for medical professionals.

Key Features:
- **Intuitive Interface:**
    PathoSync prioritizes accessibility, providing a user-friendly interface that empowers pathologists with varying technical expertise.
- **Image Upload & Preprocessing**
   **Upload Interface:** PathoSync features a user-friendly image upload interface for seamless transfer of medical images.
   **File Handling:**  Supports various image file types, including png, jpg, jpeg, tif and svs.
   **Standard Resolution:** Images are standardized to a resolution of 800x600 pixels.
   **De-noising:** Includes de-noising algorithms to improve image clarity.
   **Contrast Enhancement:** Enhances image contrast for improved visualization.
   **Image Resizing:** Resizes images for standardized processing across the platform.
   **Color Normalization:** Employs color normalization techniques for consistent color representation.
- **Cellular Annotation**
  **NuClick:** Incorporates NuClick for point annotations and labeling at the cellular level.
  **SAM:** Utilizes SAm for bounding box annotations and labeling, enhancing cellular annotation accuracy.
- **Tissue Annotation**
   - Annotations in the form of rectangles, ellipses, lines, polygons, and freehand drawings.
   - Selectable, resizable, movable annotations for different tissue structures.
   -  Customizable Colors: Personalize annotation colors for better visualization.
   -  abels and List of Labels: Allows labeling of tissue annotations with an associated list of labels.
- **WSI Upload & Preprocessing**
  **Image Upload:** Facilitates the upload of Whole Slide Images (WSI) for comprehensive pathology analysis.
  **Preprocessing:** Streamlines WSI preprocessing by efficiently extracting tiles for detailed examination.
- **Visualization**
  **Visual Magnification:** Offers a visual magnification feature for detailed views of cellular and tissue structures.
  **Tissue Masking:** Provides tissue masking capabilities for enhanced visibility and focus during the annotation process.


## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Endpoints](#endpoints)
- [Examples](#examples)
- [Built With](#built-with)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Getting Started
# PathoSync: Enhancing Pathology with Synchronized AI

## Key Features

### Intuitive Interface
PathoSync prioritizes accessibility, providing a user-friendly interface that empowers pathologists with varying technical expertise.

### Image Upload and Preprocessing
- **Upload Interface:** PathoSync features a user-friendly image upload interface for seamless transfer of medical images.
- **File Handling:** Supports various image file types, including png, jpg, jpeg, and gif.
- **Standard Resolution:** Images are standardized to a resolution of 800x600 pixels.

### Cellular Annotation
- **NuClick:** Incorporates NuClick for point annotations and labeling at the cellular level.
- **SAm:** Utilizes SAm for bounding box annotations and labeling, enhancing cellular annotation accuracy.

### Tissue Annotation
- Annotations in the form of rectangles, ellipses, lines, polygons, and freehand drawings.
- Selectable, resizable, movable annotations for different tissue structures.
- Customizable Colors: Personalize annotation colors for better visualization.
- Labels and List of Labels: Allows labeling of tissue annotations with an associated list of labels.

### WSI Upload & Preprocessing
- **Image Upload:** Facilitates the upload of Whole Slide Images (WSI) for comprehensive pathology analysis.
- **Preprocessing:** Streamlines WSI preprocessing by efficiently extracting tiles for detailed examination.

### Visualization
- **Visual Magnification:** Offers a visual magnification feature for detailed views of cellular and tissue structures.
- **Tissue Masking:** Provides tissue masking capabilities for enhanced visibility and focus during the annotation process.

### Image Preprocessing
- **De-noising:** Includes de-noising algorithms to improve image clarity.
- **Contrast Enhancement:** Enhances image contrast for improved visualization.
- **Image Resizing:** Resizes images for standardized processing across the platform.
- **Color Normalization:** Employs color normalization techniques for consistent color representation.

## Getting Started

1. Clone the repository.
   ```bash
   git clone https://github.com/BismaHaroon/F23-061-D-PathoSync.git
   cd F23-061-D-PathoSync
   ```

2. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```
# PathoSync: Enhancing Pathology with Synchronized AI

## Key Features

### Intuitive Interface
PathoSync prioritizes accessibility, providing a user-friendly interface that empowers pathologists with varying technical expertise.

### Image Upload and Preprocessing
- **Upload Interface:** PathoSync features a user-friendly image upload interface for seamless transfer of medical images.
- **File Handling:** Supports various image file types, including png, jpg, jpeg, and gif.
- **Standard Resolution:** Images are standardized to a resolution of 800x600 pixels.

### Cellular Annotation
- **NuClick:** Incorporates NuClick for point annotations and labeling at the cellular level.
- **SAm:** Utilizes SAm for bounding box annotations and labeling, enhancing cellular annotation accuracy.

### Tissue Annotation
- Annotations in the form of rectangles, ellipses, lines, polygons, and freehand drawings.
- Selectable, resizable, movable annotations for different tissue structures.
- Customizable Colors: Personalize annotation colors for better visualization.
- Labels and List of Labels: Allows labeling of tissue annotations with an associated list of labels.

### WSI Upload & Preprocessing
- **Image Upload:** Facilitates the upload of Whole Slide Images (WSI) for comprehensive pathology analysis.
- **Preprocessing:** Streamlines WSI preprocessing by efficiently extracting tiles for detailed examination.

### Visualization
- **Visual Magnification:** Offers a visual magnification feature for detailed views of cellular and tissue structures.
- **Tissue Masking:** Provides tissue masking capabilities for enhanced visibility and focus during the annotation process.

### Image Preprocessing
- **De-noising:** Includes de-noising algorithms to improve image clarity.
- **Contrast Enhancement:** Enhances image contrast for improved visualization.
- **Image Resizing:** Resizes images for standardized processing across the platform.
- **Color Normalization:** Employs color normalization techniques for consistent color representation.

### React Frontend and Flask Backend
- **Separate Frontend and Backend:** PathoSync utilizes a modern architecture with a React frontend and Flask backend, ensuring modularity and scalability.
- **Run Frontend and Backend Separately:**
  - Frontend: Navigate to the `Pathosync-frontend` directory and run the frontend separately.
    ```bash
    cd frontend
    npm install
    npm start
    ```
  - Backend: Navigate to the `Pathosync-backend` directory and run the Flask backend separately.
    ```bash
    cd backend
    pip install -r requirements.txt
    python app.py
    ```

## Getting Started
To get started with PathoSync, follow the instructions below.

1. Clone the repository.
   ```bash
   git clone https://github.com/your-username/pathosync.git
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

2. **Annotation:**
   - Engage in cellular annotation using NuClick for precise point annotations.
   - Utilize SAm for bounding box annotations to enhance cellular annotation accuracy.
   - Perform tissue annotations with customizable rectangles, ellipses, lines, polygons, and freehand drawings.
   - Personalize colors and add labels to enhance annotation visibility and organization.

3. **WSI Analysis:**
   - Upload Whole Slide Images for comprehensive pathology analysis.
   - Streamline preprocessing by efficiently extracting tiles for detailed examination.

4. **Visualization:**
   - Zoom in for visual magnification to inspect cellular and tissue structures in detail.
   - Apply tissue masking for enhanced visibility during the annotation process.

5. **Image Preprocessing:**
   - Benefit from de-noising algorithms for improved image clarity.
   - Enhance image contrast for better visualization.
   - Resize images and employ color normalization techniques for consistent color representation.

6. **React Frontend and Flask Backend:**
   - Run the frontend and backend separately for a modular and scalable architecture.
   - Follow provided instructions to set up and start each component.


## Examples
<img width="900" alt="image" src="https://github.com/BismaHaroon/F23-061-D-PathoSync/assets/125575282/0dcd5060-a5c8-440e-b63e-7de91ac05283">


## Built With

- [Python](https://www.python.org/) - The primary programming language used for backend development.
- [Flask](https://flask.palletsprojects.com/) - A micro web framework for building the backend of the application.
- [PyTorch](https://pytorch.org/) - A deep learning library utilized for implementing cutting-edge AI capabilities.
- [React](https://reactjs.org/) - A JavaScript library used for building the interactive and dynamic frontend.
- [scikit-image](https://scikit-image.org/) - An image processing library in Python for image preprocessing tasks.
- [histomicstk](https://digitalslidearchive.github.io/HistomicsTK/) - A toolkit for computational pathology, used for stain normalization and color manipulation.
- [NumPy](https://numpy.org/) - A fundamental package for scientific computing with Python, used for array operations.
- [OpenCV](https://opencv.org/) - An open-source computer vision and machine learning software library, used for image processing tasks.
- [Node.js](https://nodejs.org/) - A JavaScript runtime for executing JavaScript code on the server side, used for managing frontend dependencies.
- [npm](https://www.npmjs.com/) - A package manager for JavaScript, used for installing and managing frontend libraries and packages.

These technologies were carefully selected to create a robust, scalable, and efficient platform that combines the strengths of various tools and frameworks for a seamless user experience in digital pathology.





