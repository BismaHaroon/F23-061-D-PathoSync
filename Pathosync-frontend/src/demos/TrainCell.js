import React, { useState } from "react";
<<<<<<< HEAD
import tw from "twin.macro";
import styled from "styled-components";

import "./TrainCellStyles.css"; // Import CSS file
import LoadingDialog from "components/Loading/LoadingDialog";
import SuccessDialog from "components/Loading/SuccessDialog"; 
import ErrorDialog from "components/Loading/ErrorDialog";
import unetImage from "./unet.png";
import resnetImage from "./resnet.png";

const ModelOptionsContainer = styled.div`
  display: flex;
  justify-content: space-around; /* Adjust as needed */
  margin-bottom: 10px;
`;
const ModelOptionCard = styled.div`
  /* Card styles */
  background-color: #ffffff;
  border: 1px solid #e5e5e5;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.1);
  transition: transform 0.3s ease, box-shadow 0.3s ease; /* Add transition for hover effect */
  cursor: pointer;
  width: 300px;
  max-width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;

  /* Image container styles */
  .image-container {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    overflow: hidden;
    margin-bottom: 20px;

    img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
  }

  /* Description styles */
  p {
    font-size: 16px;
    color: #333333;
    text-align: center;
    font-weight: bold;
    margin: 0;
  }

  /* Hover effect */
  &:hover {
    transform: translateY(-5px); /* Move the card up slightly on hover */
    box-shadow: 0px 12px 24px rgba(0, 0, 0, 0.2); /* Add a stronger shadow effect */
  }
`;
const PageContainer = styled.div`
  ${tw`my-8 mx-auto max-w-4xl`}
`;
/* 
const FileInputContainer = styled.div`
    background-color: #ffff;
    padding-top: 20px;
    padding-left:20px;
    padding-right:20px;
    padding-bottom:10px;
    margin-bottom: 20px;
    
  ${tw`border border-gray-300 rounded-lg`}
`; */

const ArrowIcon = styled.svg`
  fill: white;
  height: 1em;
`;

const ModelSelect = styled.div`
  ${tw`w-full p-3 border border-gray-300 rounded-lg text-gray-600`}
`;

const ClassInput = styled.input`
  ${tw`w-full p-3 border border-gray-300 rounded-lg text-gray-600 mb-4`}
`;

const ImageInput = styled.input`
  ${tw`w-full p-3 border border-gray-300 rounded-lg text-gray-600 mb-4`}
`;

const FormContainer = styled.div`
  ${tw`p-6 mt-6 rounded-lg shadow-lg`}
  background-color: #e6e6fa; /* Light purple background */
  border: 1px solid #ccc; /* Gray border */
  box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.6); /* Shadow */
`;

const FormSection = styled.div`
  ${tw`mb-4`}
  background-color: #f9f9f9; /* Light gray background */
  border: 1px solid #ddd; /* Light gray border */
  padding: 20px; /* Add padding to the section */
  border-radius: 8px; /* Add border radius */
`;

const FileInputContainer = styled.div`
  background-color: #ffffff; /* White background */
  padding: 20px;
  margin-bottom: 20px;
  border: 1px solid #ccc; /* Gray border */
  border-radius: 8px; /* Add border radius */
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  max-height: 300px;
  overflow: auto;
  align-items: flex-start;
`;

const FormTitle = styled.h3`
  ${tw`text-lg font-semibold text-gray-800 mb-2 mt-3`}
  text-align: left;
`;



const FileInputWrapper = styled.div`
  flex: 1; /* Let the input take up remaining space */
  
`;

const FileUploadForm = styled.form`
  width: fit-content;
  height: fit-content;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const RemoveButton = styled.button`
  ${tw`flex cursor-pointer items-center justify-center text-3xl text-white caret-transparent`}
  position: absolute;
  top: 5px; /* Adjust top position */
  right: 5px; /* Adjust right position */
  width: 24px; /* Set width */
  height: 24px; /* Set height */
  border: none; /* Remove border */
  background: transparent; /* Make background transparent */
  &:hover {
    ${tw`bg-[#d1b98a]`}
    svg path {
      fill: #ffffff; /* Change fill color on hover */
    }
  }
`;


const FileInput = styled.input`
  width: 100%; /* Input takes up full width of its container */
`;
const FileInputButton = styled.label`
  ${tw`p-2 border border-gray-300 rounded-lg text-gray-600 cursor-pointer`}
  margin: 20px auto; /* Center the button horizontally and add margin */
  display: block; /* Change display to block to take full width */
  width: fit-content; /* Adjust width to fit content */
  max-width: 200px;
  text-align: center;
  background-color: #f0f0f0;
  &:hover {
    background-color: #e0e0e0;
  }
`;
const ImagePreview = styled.img`
  width: 100%;
  height: auto;
  max-height: 150px;
  border: 1px solid #ddd; /* Add a border */
  border-radius: 4px; /* Optional: Add border radius for a softer look */
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Add a subtle box shadow */
  transition: box-shadow 0.3s ease; /* Add a transition for hover effect */
  &:hover {
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3); /* Darker box shadow on hover */
  }
`;
const DeleteButton = styled.button`
   ${tw`p-2 border border-gray-300 rounded-lg text-gray-600 cursor-pointer`}
  margin: 20px 10px;
  width: 50%;
  max-width: 200px;
  text-align: center;
  background-color: #f0f0f0;
  transition: background-color 0.3s; /* Added transition for smoother hover effect */
  &:hover {
    background-color: #ddd; /* Lighter shade on hover */
  }
`;

const SubmitButton = styled.button`
  width: 200px;
  height: 40px;
  border-radius: 30px;
  border: none;
  box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.13);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  font-weight: 600;
  cursor: pointer;
  color: white;
  background: linear-gradient(to left, #6415FF, #91BDFE);
  letter-spacing: 0.7px;
  transition: transform 0.3s ease;
  margin-left:600px;
  
  &:hover {
    transform: scale(0.97);
  }
`;

// const AddClassButton = styled.button`
//   ${tw`bg-green-500 text-white font-bold py-2 px-4 rounded`}
//   &:hover {
//     ${tw`bg-green-600`}
//   }
// `;

const TrainButton = styled.button`
  ${tw`bg-yellow-500 text-white font-bold py-2 px-4 rounded`}
  &:hover {
    ${tw`bg-yellow-600`}
  }
`;

const EpochSelect = styled.select`
  ${tw`w-full p-3 border border-gray-300 rounded-lg text-gray-600 mb-4`}
`;

const LearningRateInput = styled.input`
  ${tw`w-full p-3 border border-gray-300 rounded-lg text-gray-600 mb-4`}
`;

const BatchSizeInput = styled.input`
  ${tw`w-full p-3 border border-gray-300 rounded-lg text-gray-600 mb-4`}
`;



=======
import "./TrainCellStyles.css";
>>>>>>> 8655ceccc37e8fd8d0bdcbd17d190dc036418d41

const TrainCell = () => {
  const [selectedModel, setSelectedModel] = useState(""); 
  const [showCustomClasses, setShowCustomClasses] = useState(false); 
<<<<<<< HEAD
  const [images, setImages] = useState([]); 
  const [masks, setMasks] = useState([]); 
  const [datasetName, setDatasetName] = useState(""); 
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false); 
  const [epochs, setEpochs] = useState(5); // Set a default value for epochs
  const [learningRate, setLearningRate] = useState(); 
  const [batchSize, setBatchSize] = useState(); 
  const [isLoading, setIsLoading] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [isError, setIsError] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState('idle'); // 'idle', 'in_progress', 'completed', 'failed'
  const [imagePreviews, setImagePreviews] = useState([]);
  const [imagePreviewsMask, setImagePreviewsMask] = useState([]);
  const [isRetraining, setIsRetraining] = useState(false);
  const [trainingMetrics, setTrainingMetrics] = useState({ loss: 0, accuracy: 0 });
  // Function to handle model selection
  const handleModelSelect = (model) => {
    setSelectedModel(model);
    setShowCustomClasses(model === "unet" || model ==="stardist");
  };
  const OnRetrain = () => {
    // Reset input fields and other necessary state
   
    setShowAdvancedOptions(false);
    setEpochs(5);
    setLearningRate(undefined);
    setBatchSize(undefined);
    setIsRetraining(true);
    setIsSuccess(false);
  };
const onDone= () => {
  setImages([]);
  setMasks([]);
  setDatasetName("");
  setShowAdvancedOptions(false);
  setEpochs(5);
  setLearningRate(undefined);
  setBatchSize(undefined);
  setIsLoading(false);
  setIsSuccess(false);
  setIsError(false);
  setTrainingStatus('idle');
  setImagePreviews([]);
  setImagePreviewsMask([]);
  setIsRetraining(false);
  setTrainingMetrics({ loss: 0, accuracy: 0 })

}
// Function to handle removing an uploaded image
const handleRemoveImage = (indexToRemove) => {
  setImagePreviews((prevPreviews) =>
    prevPreviews.filter((_, index) => index !== indexToRemove)
  );

  setImages((prevImages) =>
    prevImages.filter((_, index) => index !== indexToRemove)
  );
};

const handleRemoveImageMask = (indexToRemove) => {
  setImagePreviewsMask((prevPreviews) =>
    prevPreviews.filter((_, index) => index !== indexToRemove)
  );

  setImages((prevImages) =>
    prevImages.filter((_, index) => index !== indexToRemove)
  );
};


  // Function to handle uploading images

const handleImageUpload = (e) => {
  const uploadedFiles = Array.from(e.target.files);
  setImages((prevImages) => [...prevImages, ...uploadedFiles]);
  // Create image previews
  const previews = uploadedFiles.map((file) => URL.createObjectURL(file));
  setImagePreviews((prevPreviews) => [...prevPreviews, ...previews]);
};

// Function to handle uploading masks
const handleMaskUpload = (e) => {
  const uploadedFiles = Array.from(e.target.files);
  setMasks((prevMasks) => [...prevMasks, ...uploadedFiles]);
  // Create image previews
  const previews = uploadedFiles.map((file) => URL.createObjectURL(file));
  setImagePreviewsMask((prevPreviews) => [...prevPreviews, ...previews]);
};
=======
  const [images, setImages] = useState([]); // State to store uploaded image files
  const [masks, setMasks] = useState([]); // State to store uploaded mask files
  const [datasetName, setDatasetName] = useState(""); // State to store the dataset name
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false); // State to toggle showing advanced options
  const [epochs, setEpochs] = useState(10); // State to store the number of epochs
  const [learningRate, setLearningRate] = useState(0.0001); // State to store the learning rate
  const [batchSize, setBatchSize] = useState(16); // State to store the batch size

  // Function to handle model selection
  const handleModelSelect = (e) => {
    const model = e.target.value;
    setSelectedModel(model);
    setShowCustomClasses(model === "resnet"); 
  };

  // Function to handle uploading images
  const handleImageUpload = (e) => {
    setImages(e.target.files);
  };

  // Function to handle uploading masks
  const handleMaskUpload = (e) => {
    setMasks(e.target.files);
  };
>>>>>>> 8655ceccc37e8fd8d0bdcbd17d190dc036418d41

// Function to handle form submission
const handleSubmit = async (e) => {
  e.preventDefault();

  const formData = new FormData();
  formData.append("dataset_name", datasetName); // Add the dataset name to the form data

  for (let i = 0; i < images.length; i++) {
    formData.append("images", images[i]); // Append each image to the form data
    formData.append("class_names", "images"); // Append the class name "images"
  }

  for (let i = 0; i < masks.length; i++) {
    formData.append("masks", masks[i]); // Append each mask to the form data
    formData.append("class_names", "masks"); // Append the class name "masks"
  }

  try {
    const response = await fetch("http://127.0.0.1:5000/upload_with_class_cell", {
      method: "POST",
      body: formData,
    });

    if (response.ok) {
      console.log("Images and masks uploaded successfully");
      setShowAdvancedOptions(true); // Show advanced options form after successful upload
    } else {
      console.error("Failed to upload images and masks");
    }
  } catch (error) {
    console.error("Error uploading images and masks:", error);
  }
};

  // Function to handle input change for dataset name
  const handleDatasetNameChange = (e) => {
    setDatasetName(e.target.value);
  };

<<<<<<< HEAD

  // Function to handle input change for number of epochs
  const handleEpochsChange = (e) => {
    const value = parseInt(e.target.value);
    if (!isNaN(value)) {
      setEpochs(value); // Set epochs only if it's a valid integer
    }
=======
  // Function to handle input change for number of epochs
  const handleEpochsChange = (e) => {
    setEpochs(parseInt(e.target.value));
>>>>>>> 8655ceccc37e8fd8d0bdcbd17d190dc036418d41
  };

  // Function to handle input change for learning rate
  const handleLearningRateChange = (e) => {
    setLearningRate(parseFloat(e.target.value));
  };

  // Function to handle input change for batch size
  const handleBatchSizeChange = (e) => {
    setBatchSize(parseInt(e.target.value));
  };
<<<<<<< HEAD
  const pollTrainingStatus = () => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch("http://127.0.0.1:5000/check_training_status");
        const data = await response.json();
        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(interval);
          setTrainingStatus(data.status);
          setIsLoading(false);
          data.status === 'completed' ? setIsSuccess(true) : setIsError(true);
        }
        // For 'in_progress', keep polling until status changes
      } catch (error) {
        console.error("Error fetching training status:", error);
        clearInterval(interval);
        setTrainingStatus('failed');
        setIsLoading(false);
        setIsError(true);
      }
    }, 5000); // Poll every 5 seconds
  };

// Function to handle advanced options form submission (training)
const handleAdvancedOptionsSubmit = async (e) => {
  e.preventDefault();
  setIsLoading(true);
    setIsSuccess(false);
    setIsError(false);
    pollTrainingStatus();

  // Validate hyperparameters
  if (!epochs || !learningRate || !batchSize) {
    console.error("Please provide values for all hyperparameters");
    return;
  }

  // Perform model training with advanced options
  console.log("Model training started with the following advanced options:");
  console.log("Epochs:", epochs);
  console.log("Learning Rate:", learningRate);
  console.log("Batch Size:", batchSize);
  

  // Make a POST request to trigger model training
  try {
    const formData = new FormData();
    formData.append("epochs", epochs);
    formData.append("learning_rate", learningRate);
    formData.append("batch_size", batchSize);
    formData.append("modelName", selectedModel); 

    const response = await fetch("http://127.0.0.1:5000/train_cell", {
      method: "POST",
      body: formData,
    });
    console.log("Response from server:", response);
    if (response.ok) {
      const data = await response.json();
      console.log("Model training initiated successfully");

      setIsSuccess(true);

      // Display training metrics
      if (data && data.training_metrics) {
        const { loss, accuracy } = data.training_metrics;
        console.log("Training Loss:", loss);
        console.log("Training Accuracy:", accuracy);
        setTrainingMetrics({ loss, accuracy }); 

       
       
      }
    } else {
      console.error("Failed to initiate model training");
      setIsError(true);

    }
  } catch (error) {
    console.error("Error initiating model training:", error);
    setIsError(true);
  }
  finally {
    setIsLoading(false); // Stop loading after response is received
  }
};

const handleDragOver = (e) => {
  e.preventDefault();
};

const handleDrop = (e) => {
  e.preventDefault();
  const files = Array.from(e.dataTransfer.files);
  const allFiles = [...images, ...files];
  setImages(allFiles);

  const previews = allFiles.map((file) => URL.createObjectURL(file));
  setImagePreviews(previews);
};

const UNetModelOption = ({ onClick }) => {
  return (
    <ModelOptionCard onClick={onClick}>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <img src={unetImage} alt="UNet" style={{ width: '100px', height: '100px', marginBottom: '10px' }} />
        <p style={{ textAlign: 'center', fontSize: '16px', color: '#333333', fontWeight: 'bold' }}>UNet: A powerful convolutional neural network architecture designed for semantic segmentation tasks.</p>
      </div>
    </ModelOptionCard>
  );
};

const StarDistModelOption = ({ onClick }) => {
  return (
    <ModelOptionCard onClick={onClick}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <img src={resnetImage} alt="UNet" style={{ width: '100px', height: '100px', marginBottom: '10px' }} />
        <p style={{ textAlign: 'center', fontSize: '16px', color: '#333333', fontWeight: 'bold' }}>StarDist: A deep learning based cell detection and segmentation model </p>
      </div>
    </ModelOptionCard>
  );
};

  return (
    <>
    
      <PageContainer>
      <h1 className="title">Cell Detection</h1>
        <FormContainer>
        <ModelSelect>
          <ModelOptionsContainer>
            <UNetModelOption onClick={() => handleModelSelect("unet")} />
            <StarDistModelOption onClick={() => handleModelSelect("stardist")} />
            </ModelOptionsContainer>
          </ModelSelect>

          {showCustomClasses && (
            <FormSection>
              <FormTitle>Dataset Details</FormTitle>
              <form onSubmit={handleSubmit}>
                <ClassInput
=======

  // Function to handle advanced options form submission (training)
  // Function to handle advanced options form submission (training)
  const handleAdvancedOptionsSubmit = async (e) => {
    e.preventDefault();

    // Perform model training with advanced options
    console.log("Model training started with the following advanced options:");
    console.log("Epochs:", epochs);
    console.log("Learning Rate:", learningRate);
    console.log("Batch Size:", batchSize);

    // Make a POST request to trigger model training
    try {
      const response = await fetch("http://127.0.0.1:5000/train_cell", {
        method: "POST",
      });

      if (response.ok) {
        console.log("Model training initiated successfully");
      } else {
        console.error("Failed to initiate model training");
      }
    } catch (error) {
      console.error("Error initiating model training:", error);
    }
  };

  return (
    <>
      <div className="page-container">
        <h1 className="title">Cell Detection</h1>
        <div className="form-container">
          <h2 className="subtitle">Model Architecture</h2>
          <select
            className="model-select"
            value={selectedModel}
            onChange={handleModelSelect}
          >
            <option value="">Select Model Architecture</option>
            <option value="resnet">ResNet</option>
            <option value="cnn">CNN</option>
          </select>
          {showCustomClasses && (
            <div className="form-section">
              <h3 className="form-title">Upload Images and Masks</h3>
              <form className="upload-form" onSubmit={handleSubmit}>
                <input
                  className="class-input"
>>>>>>> 8655ceccc37e8fd8d0bdcbd17d190dc036418d41
                  type="text"
                  placeholder="Enter Dataset Name"
                  value={datasetName}
                  onChange={handleDatasetNameChange}
                />
<<<<<<< HEAD
                <FormTitle>Upload Cell Image Files</FormTitle>
                <FileInputButton>
  Choose Images
  <input
    type="file"
    multiple
    onChange={handleImageUpload}
    style={{ display: "none" }}
  />
</FileInputButton>
                <FileInputContainer onDragOver={handleDragOver} onDrop={handleDrop}>
                <FileInputWrapper>
               
                </FileInputWrapper>
                {imagePreviews.map((preview, index) => (
                  <div key={index} style={{ position: "relative" }}>
                    <ImagePreview src={preview} alt={`Image ${index}`} />
                    <RemoveButton onClick={() => handleRemoveImage(index)}>
                    <svg
                height="40px"
                width="40px"
                viewBox="0 0 72 72"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  fill="black"
                  d="M 19 15 C 17.977 15 16.951875 15.390875 16.171875 16.171875 C 14.609875 17.733875 14.609875 20.266125 16.171875 21.828125 L 30.34375 36 L 16.171875 50.171875 C 14.609875 51.733875 14.609875 54.266125 16.171875 55.828125 C 16.951875 56.608125 17.977 57 19 57 C 20.023 57 21.048125 56.609125 21.828125 55.828125 L 36 41.65625 L 50.171875 55.828125 C 51.731875 57.390125 54.267125 57.390125 55.828125 55.828125 C 57.391125 54.265125 57.391125 51.734875 55.828125 50.171875 L 41.65625 36 L 55.828125 21.828125 C 57.390125 20.266125 57.390125 17.733875 55.828125 16.171875 C 54.268125 14.610875 51.731875 14.609875 50.171875 16.171875 L 36 30.34375 L 21.828125 16.171875 C 21.048125 15.391875 20.023 15 19 15 z"
                ></path>
              </svg>
                    </RemoveButton>
                  </div>
                ))}
              </FileInputContainer>
              
                <FormTitle>Upload Cell Mask Files</FormTitle>
                <FileInputButton>
  Choose Images
  <input
    type="file"
    multiple
    onChange={handleMaskUpload}
    style={{ display: "none" }}
  />
</FileInputButton>
                <FileInputContainer onDragOver={handleDragOver} onDrop={handleDrop}>
                <FileInputWrapper>
                
                </FileInputWrapper>
                {imagePreviewsMask.map((preview, index) => (
                  <div key={index} style={{ position: "relative" }}>
                    <ImagePreview src={preview} alt={`Image ${index}`} />
                    <RemoveButton onClick={() => handleRemoveImageMask(index)}>
                    <svg
                height="40px"
                width="40px"
                viewBox="0 0 72 72"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  fill="black"
                  d="M 19 15 C 17.977 15 16.951875 15.390875 16.171875 16.171875 C 14.609875 17.733875 14.609875 20.266125 16.171875 21.828125 L 30.34375 36 L 16.171875 50.171875 C 14.609875 51.733875 14.609875 54.266125 16.171875 55.828125 C 16.951875 56.608125 17.977 57 19 57 C 20.023 57 21.048125 56.609125 21.828125 55.828125 L 36 41.65625 L 50.171875 55.828125 C 51.731875 57.390125 54.267125 57.390125 55.828125 55.828125 C 57.391125 54.265125 57.391125 51.734875 55.828125 50.171875 L 41.65625 36 L 55.828125 21.828125 C 57.390125 20.266125 57.390125 17.733875 55.828125 16.171875 C 54.268125 14.610875 51.731875 14.609875 50.171875 16.171875 L 36 30.34375 L 21.828125 16.171875 C 21.048125 15.391875 20.023 15 19 15 z"
                ></path>
              </svg>
                    </RemoveButton>
                  </div>
                ))}
              </FileInputContainer>
                <SubmitButton type="submit">Upload & Submit
                <ArrowIcon viewBox="0 0 448 512" height="1em">
        <path d="M438.6 278.6c12.5-12.5 12.5-32.8 0-45.3l-160-160c-12.5-12.5-32.8-12.5-45.3 0s-12.5 32.8 0 45.3L338.8 224 32 224c-17.7 0-32 14.3-32 32s14.3 32 32 32l306.7 0L233.4 393.4c-12.5 12.5-12.5 32.8 0 45.3s32.8 12.5 45.3 0l160-160z"></path>
      </ArrowIcon>
                </SubmitButton>
              </form>
            </FormSection>
          )}
          {showAdvancedOptions && (
            <FormSection>
              <FormTitle>Advanced Options</FormTitle>
              <form onSubmit={handleAdvancedOptionsSubmit}>
                <EpochSelect value={epochs} onChange={handleEpochsChange}>
                  {[5, 6, 7, 8, 9, 10].map((epoch) => (
                    <option key={epoch} value={epoch}>{epoch}</option>
                  ))}
                </EpochSelect>
                <LearningRateInput
=======
                <input
                  className="image-input"
                  type="file"
                  multiple
                  onChange={handleImageUpload}
                />
                <input
                  className="image-input"
                  type="file"
                  multiple
                  onChange={handleMaskUpload}
                />
                <button className="submit-button" type="submit">
                  Upload & Submit
                </button>
              </form>
            </div>
          )}
          {showAdvancedOptions && (
            <div className="form-section">
              <h3 className="form-title">Advanced Options</h3>
              <form onSubmit={handleAdvancedOptionsSubmit}>
                <select
                  className="epoch-select"
                  value={epochs}
                  onChange={handleEpochsChange}
                >
                  {[5, 6, 7, 8, 9, 10].map((epoch) => (
                    <option key={epoch} value={epoch}>
                      {epoch}
                    </option>
                  ))}
                </select>
                <input
                  className="learning-rate-input"
>>>>>>> 8655ceccc37e8fd8d0bdcbd17d190dc036418d41
                  type="number"
                  step="0.0001"
                  placeholder="Learning Rate"
                  value={learningRate}
                  onChange={handleLearningRateChange}
                />
<<<<<<< HEAD
                <BatchSizeInput
=======
                <input
                  className="batch-size-input"
>>>>>>> 8655ceccc37e8fd8d0bdcbd17d190dc036418d41
                  type="number"
                  placeholder="Batch Size"
                  value={batchSize}
                  onChange={handleBatchSizeChange}
                />
<<<<<<< HEAD
                <TrainButton type="submit">Train</TrainButton>
              </form>
            </FormSection>
          )}
        </FormContainer>
        {isLoading && <LoadingDialog message="Model training in progress..." />}
        {isSuccess && (
  <>
    <SuccessDialog
      message="Training successful!"
      onDone={onDone}
      trainingMetrics={trainingMetrics}
      onRetrain={OnRetrain} 
      

    />
    
  </>
)}
      {isError && <ErrorDialog message="An error occurred during training." onClose={() => setIsError(false)} />}
      </PageContainer>
      
    </>
  );
};

export default TrainCell;
=======
                <button className="train-button" type="submit">
                  Train
                </button>
              </form>
            </div>
          )}
        </div>
      </div>
    </>
  );
};
export default TrainCell;

>>>>>>> 8655ceccc37e8fd8d0bdcbd17d190dc036418d41
