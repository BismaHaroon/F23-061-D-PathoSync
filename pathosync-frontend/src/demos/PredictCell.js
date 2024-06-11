import React, { useState, useRef, useEffect } from 'react';
import tw from 'twin.macro';
import { useNavigate } from 'react-router-dom';
import LoadingDialog from 'components/Loading/LoadingDialog';
import Header from 'components/headers/light';
import Footer from 'components/footers/FiveColumnWithBackground';
import unetImage from "./unet.png";
import resnetImage from "./resnet.png";
import styled from "styled-components";

// Styled components using twin.macro with updated styles
const Container = tw.div`flex flex-col items-center justify-center min-h-screen p-8 bg-gray-100`;
const UploadWrapper = tw.div`mb-8 text-center`;
const UploadArea = tw.div`border-2 border-dashed border-gray-300 rounded-lg p-8 bg-white shadow-lg hover:border-gray-400 cursor-pointer`;
const UploadIcon = tw.span`inline-block text-xl text-gray-500`;
const Button = tw.button`mt-4 text-white px-8 py-2 rounded cursor-pointer transition duration-300 ease-in-out transform hover:scale-105 focus:outline-none bg-purple-500 hover:bg-purple-600 mb-8`;
const Typography = tw.p`mb-4 text-lg text-gray-600 text-center`;
const ImageWrapper = tw.div`mb-8 flex flex-col items-center`;
const ImageBox = tw.div`border border-purple-500 rounded p-4 max-w-md w-full mx-auto`;
const ImagePreview = tw.img`max-w-full h-auto rounded shadow-lg`;
const Heading = tw.h1`text-5xl font-bold mb-8 text-transparent bg-gradient-to-r from-purple-500 to-blue-500 bg-clip-text`;
const GridContainer = tw.div`flex flex-wrap justify-center mx-auto`;
const OverlayContainer = tw.div`w-full mt-8 flex justify-center items-center`;
const ResultImage = tw.img`max-w-xs h-auto rounded shadow-lg m-1`;

const ModelSlider = ({ modelsList, foldersList, selectedModel, handleModelSelect }) => {
  return (
    <Section>
      <Typography tw="text-2xl font-semibold mb-4 text-center">Select a Model</Typography>
      <div style={{ display: 'flex', overflowX: 'auto', maxWidth: '100%', padding: '20px' }}>
        {/* Display folders */}
        {foldersList.map((folder, index) => (
          <ModelOptionCard key={folder} onClick={() => handleModelSelect(folder)} isSelected={folder === selectedModel}>
            <ImageContainer>
              <img src={index % 2 !== 0 ? resnetImage : unetImage} alt={folder} />
            </ImageContainer>
            <ModelName>{folder}</ModelName>
          </ModelOptionCard>
        ))}

        {/* Display models */}
        {modelsList.map((model, index) => (
          <ModelOptionCard key={model} onClick={() => handleModelSelect(model)} isSelected={model === selectedModel}>
            <ImageContainer>
              <img src={index % 2 === 0 ? resnetImage : unetImage} alt={model} />
            </ImageContainer>
            <ModelName>{model}</ModelName>
          </ModelOptionCard>
        ))}
      </div>
    </Section>
  );
};

const Section = styled.div`
  ${tw`w-full max-w-4xl p-8 mb-8 bg-white rounded-lg shadow-lg`}
  &:not(:last-child) {
    ${tw`mb-8`}
  }
`;

const ImageContainer = styled.div`
  width: 120px;
  height: 120px;
  border-radius: 50%;
  overflow: hidden;
  margin: 0 auto 20px;
  display: flex;
  justify-content: center;
  align-items: center;

  img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
  }
`;

const ModelName = styled.p`
  font-size: 16px;
  color: #333333;
  text-align: center;
  font-weight: bold;
  margin: 0;
`;

const ModelOptionCard = styled.div`
  background-color: #ffffff;
  border: 1px solid #e5e5e5;
  border-radius: 12px;
  padding: 10px;
  box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.1);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  cursor: pointer;
  width: 220px;
  max-width: 100%;
  flex: 0 0 auto;
  margin-right: 20px;

  ${({ isSelected }) =>
    isSelected &&
    `
    border-color: #7c3aed;
    box-shadow: 0px 12px 24px rgba(124, 58, 237, 0.2);
  `}

  &:hover {
    transform: translateY(-5px);
    box-shadow: 0px 12px 24px rgba(0, 0, 0, 0.2);
  }
`;

const PredictCell = () => {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState("");
  const [modelsList, setModelsList] = useState([]);
  const [foldersList, setFoldersList] = useState([]);
  const [resultData, setResultData] = useState(null);
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const resultRef = useRef(null);

  useEffect(() => {
    // Fetch models list from backend when component mounts
    const fetchModelsList = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5000/modelsList");
        if (response.ok) {
          const data = await response.json();
          setModelsList(data.models);
          setFoldersList(data.folders);
        } else {
          console.error("Failed to fetch models list");
        }
      } catch (error) {
        console.error("Error fetching models list:", error);
      }
    };

    fetchModelsList();
  }, []);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setUploadedImage(file);
      setResultData(null); // Clear previous result data
    }
  };

  const handleConfirm = async () => {
    if (uploadedImage && selectedModel) {
      setIsLoading(true);

      try {
        const formData = new FormData();
        formData.append("image", uploadedImage);
        formData.append("model", selectedModel);

        const response = await fetch("http://127.0.0.1:5000/CellPredict", {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          const data = await response.json();
          setResultData(data);
          setTimeout(() => {
            resultRef.current.scrollIntoView({ behavior: 'smooth' });
          }, 100); // Adjust the timeout if necessary
        } else {
          console.error("Failed to predict");
        }
      } catch (error) {
        console.error("Error during image processing:", error);
      } finally {
        setIsLoading(false);
      }
    } else {
      console.log("Please upload an image and select a model");
    }
  };

  const handleModelSelect = (model) => {
    setSelectedModel(model);
  };

  return (
    <>
      <Header />
      <Container>
        <Heading>Detect and Count Cells</Heading>

        <ModelSlider modelsList={modelsList} foldersList={foldersList} selectedModel={selectedModel} handleModelSelect={handleModelSelect} />

        <Section>
          <Typography tw="text-2xl font-semibold mb-4 text-center">Upload Your Image</Typography>
          <UploadWrapper>
            <UploadArea onClick={() => fileInputRef.current.click()}>
              <UploadIcon>📤</UploadIcon>
              <p>Drag and drop your image here, or click to select a file to upload.</p>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                css={tw`hidden`}
              />
            </UploadArea>
          </UploadWrapper>

          {uploadedImage && (
            <ImageWrapper>
              <ImageBox>
                <p css={tw`text-lg font-bold`}>Uploaded Image:</p>
                <ImagePreview src={URL.createObjectURL(uploadedImage)} alt="Uploaded Preview" />
              </ImageBox>
            </ImageWrapper>
          )}
        </Section>

        <Button onClick={handleConfirm} disabled={!selectedModel}>
          Predict
        </Button>

        {isLoading && <LoadingDialog message="Prediction in progress..." />}

        {resultData && (
          <Section ref={resultRef}>
            <Typography tw="text-2xl font-semibold mb-4 text-center">Prediction Result</Typography>
            <ImageWrapper>
              <ImageBox>
                <ImagePreview src={`http://127.0.0.1:5000/${resultData.segmented_image_filename}`} alt="Prediction Result" />
</ImageBox>
</ImageWrapper>
</Section>
)}
</Container>
<Footer />
</>
);
};

export default PredictCell;
