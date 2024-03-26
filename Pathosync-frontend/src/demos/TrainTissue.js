import React, { useState } from "react";
import tw from "twin.macro";
import styled from "styled-components";
import Header from "components/headers/light";
import Footer from "components/footers/FiveColumnWithBackground";
import "./TrainTissueStyles.css"; // Import CSS file



const PageContainer = styled.div`
  ${tw`my-8 mx-auto max-w-4xl`}
`;

const FileInputContainer = styled.div`
    background-color: #ffff;
    padding-top: 20px;
    padding-left:20px;
    padding-right:20px;
    padding-bottom:10px;
    margin-bottom: 20px;
    
  ${tw`border border-gray-300 rounded-lg`}
`;


const ModelSelect = styled.select`
  ${tw`w-full p-3 border border-gray-300 rounded-lg text-gray-600 mb-1`}
`;

const UploadForm = styled.form`
  ${tw`mt-4`}
`;

const ClassInput = styled.input`
  ${tw`w-full p-3 border border-gray-300 rounded-lg text-gray-600 mb-1`}
`;

const ImageInput = styled.input`
  ${tw`w-full p-3 border border-gray-300 rounded-lg text-gray-600 mb-4`}
`;

const FormContainer = styled.div`
  ${tw`p-6 mt-6 rounded-lg shadow-lg`}
  background-color: #e6e6fa ;
`;

const FormTitle = styled.h3`
  ${tw`text-lg font-semibold text-gray-800 mb-2 mt-3`}
  text-align: left;
`;

const FormSection = styled.div`
  ${tw`mb-4 `}
`;

const ButtonContainer = styled.div`
  ${tw`flex`}
`;

const SubmitButton = styled.button`
background-color: #37097d ;
  ${tw` text-white font-bold py-2 px-4 rounded`}
  &:hover {
    background-color:#9400d3 ;
  }
`;

const AddClassButton = styled.button`
background-color: #37097d ;
  ${tw` text-white font-bold py-2 px-4 rounded`}
  margin-right: 10px; /* Add margin between Add Class and Upload & Submit */
  &:hover {
    background-color:#9400d3 ;
  }
`;

const TrainButton = styled.button`
background-color: #37097d ;
  ${tw` text-white font-bold py-2 px-4 rounded`}
  &:hover {
    background-color:#9400d3
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

const CsvInput = styled.input`
  ${tw`w-full p-3 border border-gray-300 rounded-lg text-gray-600 mb-4`}
`;


const TrainTissue = () => {
  const [selectedModel, setSelectedModel] = useState(""); 
  const [showCustomClasses, setShowCustomClasses] = useState(false); 
  const [classes, setClasses] = useState([{ name: "", images: [] }]);
  const [datasetName, setDatasetName] = useState(""); // State to store the dataset name
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false); // State to toggle showing advanced options
  const [epochs, setEpochs] = useState(10); // State to store the number of epochs
  const [learningRate, setLearningRate] = useState(0.0001); // State to store the learning rate
  const [batchSize, setBatchSize] = useState(16); // State to store the batch size
  const [showDatasetFields, setShowDatasetFields] = useState(false);
  

  const [csvFiles, setCsvFiles] = useState([]);

  // Function to handle uploading CSV files
  const handleCsvUpload = (files) => {
    // Convert FileList to an array and update state
    const csvFilesArray = Array.from(files);
    setCsvFiles(csvFilesArray);
  };

// Function to handle model selection
const handleModelSelect = (e) => {
  const model = e.target.value;
  setSelectedModel(model);
  
  if (model === "cnn") {
    setShowCustomClasses(true); // Show the custom classes form when CNN is selected
    setShowDatasetFields(false); // Hide the dataset fields when CNN is selected
  } else {
    setShowCustomClasses(false); // Hide the custom classes form when ResNet is selected
    setShowDatasetFields(true); // Show the dataset fields when ResNet is selected
  }
};


  

  // Function to handle uploading images for custom classes
  const handleImageUpload = (classIndex, files) => {
    const updatedClasses = [...classes];
    updatedClasses[classIndex] = {
      ...updatedClasses[classIndex],
      images: files,
    };
    setClasses(updatedClasses);
  };

  // Function to handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();

    const formData = new FormData();
    formData.append("dataset_name", datasetName); // Add the dataset name to the form data

    for (let i = 0; i < classes.length; i++) {
      const currentClass = classes[i];
      for (let j = 0; j < currentClass.images.length; j++) {
        const currentImage = currentClass.images[j];
        formData.append("images", currentImage); // Append each image to the form data
        formData.append("class_names", currentClass.name); // Append the class name to the form data
      }
    }

    try {
      const response = await fetch("http://127.0.0.1:5000/upload_with_class", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        console.log("Images uploaded and processed successfully");
        setShowAdvancedOptions(true); // Show advanced options form after successful upload
      } else {
        console.error("Failed to upload images");
      }
    } catch (error) {
      console.error("Error uploading images:", error);
    }
  };

  // Function to add a new class
  const addClass = () => {
    setClasses([...classes, { name: "", images: [] }]);
  };

  // Function to handle input change for class name
  const handleClassNameChange= (e, classIndex) => {
    const updatedClasses = [...classes];
    updatedClasses[classIndex] = {
      ...updatedClasses[classIndex],
      name: e.target.value,
    };
    setClasses(updatedClasses);
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
  const handleAdvancedOptionsSubmit = async (e) => {
    e.preventDefault();

    // Perform model training with advanced options
    console.log("Model training started with the following advanced options:");
    console.log("Epochs:", epochs);
    console.log("Learning Rate:", learningRate);
    console.log("Batch Size:", batchSize);

    // You can make a POST request here to send the advanced options data to the backend for model training
  };

  const handleTrain = async (e) => {
    e.preventDefault();
  
    try {
      const response = await fetch("http://127.0.0.1:5000/train_tissue_cnn", {
        method: "POST",
      });
  
      if (response.ok) {
        console.log("Model training started successfully");
        // Optionally, you can handle success actions here
      } else {
        console.error("Failed to start model training");
        // Optionally, you can handle failure actions here
      }
    } catch (error) {
      console.error("Error starting model training:", error);
      // Optionally, you can handle error actions here
    }
  };

// RESNET 
// Function to handle form submission for ResNet
const handleResNetSubmit = async (e) => {
  e.preventDefault();

  // Here you can perform any client-side validation or other necessary actions

  // For now, let's just log the dataset name and CSV files (if uploaded)
  console.log("Dataset Name:", datasetName);
  console.log("CSV Files:", csvFiles);

  // Toggle the state to show advanced options
  setShowAdvancedOptions(true);
};



  return (
    <>
    <Header></Header>
      <PageContainer>
        <h1 className="title">Tissue Classification</h1>
        <FormContainer>
        <h2 className="subtitle">Model Architecture</h2>
          <ModelSelect value={selectedModel} onChange={handleModelSelect}>
            <option value="">Select Model Architecture</option>
            <option value="resnet">ResNet</option>
            <option value="cnn">CNN</option>
          </ModelSelect>

          {/* New section for handling dataset information */}
  {showDatasetFields && (
    <FormSection>
      <FormTitle>Dataset Information</FormTitle>
      <ClassInput
        type="text"
        placeholder="Enter Dataset Name"
        value={datasetName}
        onChange={handleDatasetNameChange}
      />
      <FileInputContainer>
        <FormTitle>Upload Image Files</FormTitle>
        <ImageInput
          type="file"
          multiple
          onChange={(e) => handleImageUpload(e.target.files)}
        />
      </FileInputContainer>
      <FileInputContainer>
        <FormTitle>Upload CSV File</FormTitle>
        <CsvInput
          type="file"
          multiple  // Allow multiple files to be uploaded
          onChange={(e) => handleCsvUpload(e.target.files)}
        />
      </FileInputContainer>
      <ButtonContainer>
      <SubmitButton type="submit" onClick={handleResNetSubmit}>Upload & Submit</SubmitButton>
    </ButtonContainer>
    </FormSection>
  )}


          {showCustomClasses && (
            <FormSection>
              <FormTitle>Upload Custom Classes</FormTitle>
              <UploadForm onSubmit={handleSubmit}>
                <ClassInput
                  type="text"
                  placeholder="Enter Dataset Name"
                  value={datasetName}
                  onChange={handleDatasetNameChange}
                />
                {classes.map((classData, i) => (
                  <div key={i}>
                    <FormSection>
                      <FormTitle>Class Data</FormTitle>
                      <ClassInput
                        type="text"
                        placeholder="Enter Class Name"
                        value={classData.name}
                        onChange={(e) => handleClassNameChange(e, i)}
                      />
                    </FormSection>
                    <FileInputContainer> {/* Add a container for the file input */}
                    <ImageInput
                      type="file"
                      multiple
                      onChange={(e) => handleImageUpload(i, e.target.files)}
                    />
                  </FileInputContainer>
                  </div>
                ))}
                <ButtonContainer>
              <AddClassButton onClick={addClass}>Add Class</AddClassButton>
              <SubmitButton type="submit">Upload & Submit</SubmitButton>
              </ButtonContainer>
              </UploadForm>
              
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
                  type="number"
                  step="0.0001"
                  placeholder="Learning Rate"
                  value={learningRate}
                  onChange={handleLearningRateChange}
                />
                <BatchSizeInput
                  type="number"
                  placeholder="Batch Size"
                  value={batchSize}
                  onChange={handleBatchSizeChange}
                />
                <TrainButton type="button" onClick={handleTrain}>Train</TrainButton>
              </form>
            </FormSection>
          )}
        </FormContainer>
      </PageContainer>
      <Footer></Footer>
    </>
  );
};

export default TrainTissue;
