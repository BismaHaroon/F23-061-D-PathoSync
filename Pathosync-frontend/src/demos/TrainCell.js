import React, { useState } from "react";
import "./TrainCellStyles.css";

const TrainCell = () => {
  const [selectedModel, setSelectedModel] = useState(""); 
  const [showCustomClasses, setShowCustomClasses] = useState(false); 
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

  // Function to handle input change for number of epochs
  const handleEpochsChange = (e) => {
    setEpochs(parseInt(e.target.value));
  };

  // Function to handle input change for learning rate
  const handleLearningRateChange = (e) => {
    setLearningRate(parseFloat(e.target.value));
  };

  // Function to handle input change for batch size
  const handleBatchSizeChange = (e) => {
    setBatchSize(parseInt(e.target.value));
  };

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
                  type="text"
                  placeholder="Enter Dataset Name"
                  value={datasetName}
                  onChange={handleDatasetNameChange}
                />
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
                  type="number"
                  step="0.0001"
                  placeholder="Learning Rate"
                  value={learningRate}
                  onChange={handleLearningRateChange}
                />
                <input
                  className="batch-size-input"
                  type="number"
                  placeholder="Batch Size"
                  value={batchSize}
                  onChange={handleBatchSizeChange}
                />
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

