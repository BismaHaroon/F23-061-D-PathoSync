import React, { useState } from "react";
import tw from "twin.macro";
import styled from "styled-components";
import Header from "components/headers/light";
import Footer from "components/footers/FiveColumnWithBackground";
import "./TrainTissueStyles.css"; // Import CSS file
import LoadingDialog from "components/Loading/LoadingDialog"; // Make sure this path is correct
import SuccessDialog from "components/Loading/SuccessDialogTissue"; // Make sure this path is correct
import ErrorDialog from "components/Loading/ErrorDialog";
import unetImage from "./unet.png";
import resnetImage from "./resnet.png";

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


const BinIcon = styled.svg`
  fill: black;
  height: 1em;
`;
const PageContainer = styled.div`
  ${tw`my-8 mx-auto max-w-4xl`}
`;

const FileInputContainer = styled.div`
  background-color: #ffffff;
  padding: 20px;
  margin-bottom: 20px;
  border: 1px solid #ccc;
  border-radius: 8px;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 10px;
  max-height: 300px;
  overflow: auto;
`;
const ModelSelect = styled.select`
  ${tw`w-full p-3 border border-gray-300 rounded-lg text-gray-600 mb-1`}
`;

const UploadForm = styled.form`
  ${tw`mt-4`}
`;


// const FileInputWrapper = styled.div`
//   flex: 1; /* Let the input take up remaining space */
  
// `;
const ButtonContainer1 = styled.div`
  ${tw`flex justify-center items-center`}
`;
const ClassInput = styled.input`
  ${tw`w-full p-3 border border-gray-300 rounded-lg text-gray-600 mb-1`}
`;

const ImageInputButton = styled.label`
  ${tw`p-2 border border-gray-300 rounded-lg text-gray-600 cursor-pointer`}
  margin: 20px 10px; /* Adjust margin for spacing */
  width: 50%;
  max-width: 200px;
  text-align: center;
  background-color: #f0f0f0;
  &:hover {
    background-color: #e0e0e0;
  }
`;

const FormContainer = styled.div`
  ${tw`p-6 mt-6 rounded-lg shadow-lg`}
  background-color: #e6e6fa; /* Light purple background */
  border: 1px solid #ccc; /* Gray border */
  box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.6); /* Shadow */
  `; 

const FormTitle = styled.h3`
  ${tw`text-lg font-semibold text-gray-800 mb-2 mt-3`}
  text-align: left;
`;

const FormTitle2 = styled.h4`
  ${tw`text-lg font-semibold text-gray-700 mb-1 mt-3`}
  text-align: left;
`;

const FormSection = styled.div`
  ${tw`mb-4`}
  background-color: #f9f9f9; /* Light gray background */
  border: 1px solid #ddd; /* Light gray border */
  padding: 20px; /* Add padding to the section */
  border-radius: 8px; /* Add border radius */
`;

const ButtonContainer = styled.div`
  ${tw`flex`}
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
  margin-left:400px;
  &:hover {
    transform: scale(0.97);
  }
`;

const ArrowIcon = styled.svg`
  fill: white;
  height: 1em;
`;

const AddClassButton = styled.button`
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

&:hover {
  transform: scale(0.97);
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

const ModelOptionsContainer = styled.div`
  display: flex;
  justify-content: space-around; /* Adjust as needed */
  margin-bottom: 20px;
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
  width: 250px;
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
const ImageGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  grid-gap: 10px;
`;

const ImageContainer = styled.div`
  position: relative;
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


const Title = styled.h1`
  text-align: center;
  margin-bottom: 40px;
  color: #333;
  font-size: 2.5rem;
  background: -webkit-linear-gradient(#6415FF, #91BDFE);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
`;
const ImagePreviewContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  grid-gap: 10px;
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
  const [isLoading, setIsLoading] = useState(false); // State for showing loading dialog
  const [isSuccess, setIsSuccess] = useState(false); // State for showing success dialog
  const [isError, setIsError] = useState(false); // State for showing error dialog
  const [errorMessage, setErrorMessage] = useState(""); // State for storing error message
  const [classImagePreviews, setClassImagePreviews] = useState(Array(classes.length).fill([]));
  const [classImages, setClassImages] = useState(Array(classes.length).fill([]));
  const [Accuracy, setAccuracy] = useState(null); // State for accuracy
  const [Loss, setLoss] = useState(null); 



  const [csvFiles, setCsvFiles] = useState([]);

  // Function to handle uploading CSV files
  const handleCsvUpload = (files) => {
    // Convert FileList to an array and update state
    const csvFilesArray = Array.from(files);
    setCsvFiles(csvFilesArray);
  };

// Function to handle model selection
const handleModelSelect = (model) => {
  setSelectedModel(model);
  setShowCustomClasses(true); // Show the custom classes form when CNN is selected
  setShowDatasetFields(false); // Hide the dataset fields when CNN is selected
 
};
const handleRemoveClass = (classIndex) => {
  const confirmDelete = window.confirm("Are you sure you want to delete this class?");
  if (!confirmDelete) {
    return;
  }

  const updatedClasses = [...classes];
  updatedClasses.splice(classIndex, 1);
  setClasses(updatedClasses);

  const updatedClassImagePreviews = [...classImagePreviews];
  updatedClassImagePreviews.splice(classIndex, 1);
  setClassImagePreviews(updatedClassImagePreviews);

  const updatedClassImages = [...classImages];
  updatedClassImages.splice(classIndex, 1);
  setClassImages(updatedClassImages);
};

const onDone = () => {
  // Reset input fields and other necessary state
  setDatasetName("");
  setClasses([{ name: "", images: [] }]);
  setEpochs(10);
  setLearningRate(0.0001);
  setBatchSize(16);
  setIsSuccess(false);
  setClassImagePreviews([]);
  setIsSuccess(false);
    setAccuracy(null);
    setLoss(null);
  setShowCustomClasses(false);

};

const onRetrain = () => {
  // Reset input fields and other necessary state
  setDatasetName("");
  setClasses([{ name: "", images: [] }]);
  setEpochs(10);
  setLearningRate(0.0001);
  setBatchSize(16);
  setShowAdvancedOptions(false);
  setIsSuccess(false);
};
  

const handleImageUpload = (classIndex, files) => {
  const updatedClasses = [...classes];
  updatedClasses[classIndex] = {
    ...updatedClasses[classIndex],
    images: files,
  };
  setClasses(updatedClasses);

  // Generate image previews
  const previews = Array.from(files).map((file) => URL.createObjectURL(file));
  setClassImagePreviews((prevPreviews) => {
    const updatedPreviews = [...prevPreviews];
    if (typeof updatedPreviews[classIndex] === 'undefined') {
      updatedPreviews[classIndex] = []; // Initialize as an empty array if undefined
    }
    updatedPreviews[classIndex] = [...updatedPreviews[classIndex], ...previews];
    return updatedPreviews;
  });
  // Set image files
  setClassImages((prevImages) => {
    const updatedImages = [...prevImages];
    if (typeof updatedImages[classIndex] === 'undefined') {
      updatedImages[classIndex] = []; // Initialize as an empty array if undefined
    }
    updatedImages[classIndex] = [...updatedImages[classIndex], ...files];
    return updatedImages;
  });
};



const handleRemoveImage = (classIndex, indexToRemove) => {
  setClassImagePreviews((prevPreviews) => {
    const updatedPreviews = [...prevPreviews];
    updatedPreviews[classIndex] = updatedPreviews[classIndex].filter((_, index) => index !== indexToRemove);
    return updatedPreviews;
  });

  setClassImages((prevImages) => {
    const updatedImages = [...prevImages];
    updatedImages[classIndex] = updatedImages[classIndex].filter((_, index) => index !== indexToRemove);
    return updatedImages;
  });
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
    if (selectedModel === "densenet" && classes.length >= 2) {
      alert("You can only add up to 2 classes for DenseNet model");
      return;
    }
    setClasses([...classes, { name: '', images: [] }]);
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
    setIsLoading(true);
    let trainEndpoint = " ";

    if (selectedModel === "resnet") {
      trainEndpoint = "http://127.0.0.1:5000/train_tissue_resnet";
    } else if (selectedModel=="densenet")
    {
      trainEndpoint = "http://127.0.0.1:5000/train_tissue_cnn";
    }
    // else if (selectedModel=="convnext")
    // {
    //   trainEndpoint="http://127.0.0.1:5000/train_tissue_convNext"
    // }


    try {
      const formData = new FormData();
      formData.append("epochs", epochs);
      formData.append("learning_rate", learningRate);
      formData.append("batch_size", batchSize);

      const response = await fetch(trainEndpoint, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const responseData = await response.json();
        
        setIsSuccess(true);
        console.log("Model training started successfully");
        console.log("Accuracy:", responseData.accuracy);
        console.log("Loss:", responseData.loss);
        setAccuracy(responseData.accuracy);
        setLoss(responseData.loss);

      // Optionally, you can pass accuracy and loss to the success dialog
     
      } else {
        setErrorMessage(errorMessage.errorMessage);
        console.error("Failed to start model training");
        // Optionally, you can handle failure actions here
        console.log(errorMessage)
      }
    } catch (error) {
      setErrorMessage(errorMessage.errorMessage);
      console.error("Error starting model training:", error);
      // Optionally, you can handle error actions here
    }finally {
      setIsLoading(false); // Hide loading dialog
    }

  };


  // const ConVNextModelOption = ({ onClick }) => {
  //   return (
  //     <ModelOptionCard onClick={onClick}>
  //       <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
  //         <img src={unetImage} alt="ConVNext" style={{ width: '100px', height: '100px', marginBottom: '10px' }} />
  //         <p style={{ textAlign: 'center', fontSize: '16px', color: '#333333', fontWeight: 'bold' }}>ConVNext: An advanced convolutional neural network architecture for image classification and feature extraction.</p>
  //       </div>
  //     </ModelOptionCard>
  //   );
  // };

  const DenseNetModelOption = ({ onClick }) => {
    return (
      <ModelOptionCard onClick={onClick}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <img src={unetImage} alt="densenet" style={{ width: '100px', height: '100px', marginBottom: '10px' }} />
          <p style={{ textAlign: 'center', fontSize: '16px', color: '#333333', fontWeight: 'bold' }}>DenseNetModelOption: An advanced convolutional neural network architecture for image classification and feature extraction.</p>
        </div>
      </ModelOptionCard>
    );
  };



  const ResnetModelOption = ({ onClick }) => {
    return (
      <ModelOptionCard onClick={onClick}>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <img src={resnetImage} alt="resnet" style={{ width: '100px', height: '100px', marginBottom: '10px' }} />
          <p style={{ textAlign: 'center', fontSize: '16px', color: '#333333', fontWeight: 'bold' }}>ResNet: A deep convolutional neural network known for its depth and efficiency in image classification tasks.</p>
        </div>
      </ModelOptionCard>
    );
  };
  


  return (
    <>
   
      <PageContainer>
        <Title>Tissue Classification</Title>
        <FormContainer>
        <ModelOptionsContainer>
        {/* <ConVNextModelOption onClick={() => handleModelSelect("convnet")} /> */}
        <ResnetModelOption onClick={() => handleModelSelect("resnet")} />
        <DenseNetModelOption onClick={() => handleModelSelect("densenet")} />
        </ModelOptionsContainer>

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
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <FormTitle>Class Data</FormTitle>
        
        
      </div>
      <ClassInput
        type="text"
        placeholder="Enter Class Name"
        value={classData.name}
        onChange={(e) => handleClassNameChange(e, i)}
      />
      {/* <FileInputWrapper> */}
      <ButtonContainer1>
      <ImageInputButton>
                    Choose Images
                    <input
                      type="file"
                      multiple
                      onChange={(e) => handleImageUpload(i, e.target.files)}
                      style={{ display: "none" }}
                    />
                  </ImageInputButton>
                  <DeleteButton onClick={() => handleRemoveClass(i)}> Delete Class
                      
                    </DeleteButton>
                    </ButtonContainer1>
    {/* </FileInputWrapper> */}
      {/* <RemoveButton onClick={() => handleRemoveClass(i)}>
                      <BinIcon viewBox="0 0 24 24">
                        <path d="M3 6l3 18h12l3-18h-18zm12 4v10h-2v-10h-2v10h-2v-10h-2v10h-2v-10h-2v10h-2v-10z" />
                      </BinIcon>
                    </RemoveButton> */}
    </FormSection>
    <FileInputContainer>
   
      {/* Render image previews and remove buttons */}
      {classImagePreviews[i] &&
        classImagePreviews[i].map((preview, index) => (
          <div key={index} style={{ position: "relative" }}>
            <ImagePreview src={preview} alt={`Image ${index}`} />
            <RemoveButton onClick={() => handleRemoveImage(i, index)}>
              <svg
                height="40px"
                width="40px"
                viewBox="0 0 72 72"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  fill="white"
                  d="M 19 15 C 17.977 15 16.951875 15.390875 16.171875 16.171875 C 14.609875 17.733875 14.609875 20.266125 16.171875 21.828125 L 30.34375 36 L 16.171875 50.171875 C 14.609875 51.733875 14.609875 54.266125 16.171875 55.828125 C 16.951875 56.608125 17.977 57 19 57 C 20.023 57 21.048125 56.609125 21.828125 55.828125 L 36 41.65625 L 50.171875 55.828125 C 51.731875 57.390125 54.267125 57.390125 55.828125 55.828125 C 57.391125 54.265125 57.391125 51.734875 55.828125 50.171875 L 41.65625 36 L 55.828125 21.828125 C 57.390125 20.266125 57.390125 17.733875 55.828125 16.171875 C 54.268125 14.610875 51.731875 14.609875 50.171875 16.171875 L 36 30.34375 L 21.828125 16.171875 C 21.048125 15.391875 20.023 15 19 15 z"
                ></path>
              </svg>
            </RemoveButton>
          </div>
        ))}
    </FileInputContainer>
  </div>
))}

 

                <ButtonContainer>
              <AddClassButton onClick={addClass}>Add Class                 
      </AddClassButton>
              <SubmitButton type="submit">Upload & Submit <ArrowIcon viewBox="0 0 448 512" height="1em">
        <path d="M438.6 278.6c12.5-12.5 12.5-32.8 0-45.3l-160-160c-12.5-12.5-32.8-12.5-45.3 0s-12.5 32.8 0 45.3L338.8 224 32 224c-17.7 0-32 14.3-32 32s14.3 32 32 32l306.7 0L233.4 393.4c-12.5 12.5-12.5 32.8 0 45.3s32.8 12.5 45.3 0l160-160z"></path>
      </ArrowIcon> </SubmitButton>
              </ButtonContainer>
              </UploadForm>
              
            </FormSection>
          )}
          {showAdvancedOptions && (
            <FormSection>
              <FormTitle>Advanced Options</FormTitle>
              <form onSubmit={handleAdvancedOptionsSubmit}>
                <FormTitle2>Epochs</FormTitle2>
                <EpochSelect value={epochs} onChange={handleEpochsChange}>
                  {[1,3,5, 6, 7, 8, 9, 10].map((epoch) => (
                    <option key={epoch} value={epoch}>{epoch}</option>
                  ))}
                </EpochSelect>
                <FormTitle2>Learning Rate</FormTitle2>
                <LearningRateInput
                  type="number"
                  step="0.0001"
                  placeholder="Learning Rate"
                  value={learningRate}
                  onChange={handleLearningRateChange}
                />
                <FormTitle2>Batch Size</FormTitle2>
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
      {isLoading && <LoadingDialog message="Model training in progress..." />}
      {isSuccess && (
        <SuccessDialog
          message="Training successful!"
          onDone={onDone}
          onRetrain={onRetrain}
          accuracy={Accuracy}
          loss={Loss}
        />
      )}
            {isError && <ErrorDialog message={errorMessage} onClose={() => setIsError(false)} />}
      
    </>
  );
};

export default TrainTissue;