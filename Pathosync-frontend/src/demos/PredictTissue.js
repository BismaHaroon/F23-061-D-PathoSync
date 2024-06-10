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

const OverlayContainer = tw.div`w-full mt-8 flex justify-center items-center`; 
const ResultImage = tw.img`max-w-full h-auto rounded shadow-lg m-1`;

const ModelSlider = ({ modelsList, selectedModel, handleModelSelect }) => {
  return (
    <Section>
      <Typography tw="text-2xl font-semibold mb-4 text-center">Select a Model</Typography>
      <div style={{ display: 'flex', overflowX: 'auto', maxWidth: '100%', padding: '20px' }}>
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
const GridContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  justify-items: center;
  margin-top: 20px;
`;

const CenteredSection = styled(Section)`
  text-align: center;
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

const PatchPlotGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  grid-template-rows: auto auto;
  grid-gap: 10px;
  justify-items: center;
`;

const PredictTissue = () => {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [isConfirmed, setIsConfirmed] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState("");
  const [modelsList, setModelsList] = useState([]); 
  const [patchPlots, setPatchPlots] = useState([]); 
  const [overlayPlot, setOverlayPlot] = useState(""); 
  const navigate = useNavigate();
  const fileInputRef = useRef(null);

  useEffect(() => {
    // Fetch models list from backend when component mounts
    const fetchModelsList = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5000/modelsListTissue");
        if (response.ok) {
          const data = await response.json();
          setModelsList(data.models);
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
    if (file) { // This check prevents the double prompt if the user cancels the file selection dialog
      setUploadedImage(file);
    }
  };

  const handleConfirm = async () => {
    if (uploadedImage && selectedModel) {
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append("image", uploadedImage);
      formData.append("model", selectedModel); 

      const response = await fetch("http://127.0.0.1:5000/TissuePredict", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setPatchPlots(data.patch_plot_files); 
        setOverlayPlot(data.overlay_plot);
        setIsConfirmed(true); 
        console.log("Overlay Plot File Path:", data.overlay_plot);



        
      } else {
        console.error("Failed to predict");
      }
    } catch (error) {
      console.error("Error during image processing:", error);
    } finally {
      setIsLoading(false);
    }
  };}

  const handleModelSelect = (model) => {
    setSelectedModel(model);
  };
  
  const handleClickUpload = () => {
    fileInputRef.current.click();
  };

  const triggerFileInput = () => fileInputRef.current.click();

  return (
    <>
      
      <Container>
        <Heading>Classify Tissues</Heading>
        
        <ModelSlider modelsList={modelsList} selectedModel={selectedModel} handleModelSelect={handleModelSelect} />
        <Section>
  <Typography tw="text-2xl font-semibold mb-4 text-center">Upload Your Image</Typography>
  <UploadWrapper>
    <UploadArea >
      <UploadIcon>📤</UploadIcon>
      <p>Drag and drop your image here, or click to select a file to upload.</p>
      <input
        ref={fileInputRef}
        type="file"
        id="imageInput"
        accept="image/*"
        onChange={handleImageUpload}
        css={tw`hidden`}
      />
    </UploadArea>
  </UploadWrapper>
  
  {uploadedImage && !isConfirmed && (
    <ImageWrapper>
      <ImageBox>
        <p css={tw`text-lg font-bold`}>Uploaded Image:</p>
        <ImagePreview
          src={URL.createObjectURL(uploadedImage)}
          alt="Uploaded Preview"
        />
      </ImageBox>
    </ImageWrapper>
  )}
</Section>

        <Button
          onClick={handleConfirm}
          disabled={!uploadedImage || !selectedModel}
        >
          Confirm
        </Button>

        {isLoading && <LoadingDialog message="Prediction in progress..." />}
        {isConfirmed && (
          <>
            {/* {Array.isArray(patchPlots) && patchPlots.length > 0 && (
              <CenteredSection>
                <Section>
                <Typography tw="text-2xl font-semibold mb-4">Patch Plotting</Typography>
                <GridContainer>
                <PatchPlotGrid>
                  {patchPlots.map((plotFile, index) => (
                    <div key={index} >
                      <ResultImage src={`http://127.0.0.1:5000/download_patch_plot/${plotFile}`} alt={`Patch Plot ${index}`} />
                    </div>
                  ))}
                </PatchPlotGrid>
                </GridContainer>
                </Section>
              </CenteredSection>
            )} */}
    {overlayPlot && (
      <Section>
        <Typography tw="text-2xl font-semibold mb-4 text-center">Overlay Plotting</Typography>
        <OverlayContainer>
          <a href={`http://127.0.0.1:5000/download_overlay_plot/${overlayPlot}`} download="Overlay_Plot.png">
            <ResultImage src={`http://127.0.0.1:5000/download_overlay_plot/${overlayPlot}`} alt="Overlay Plot" />
          </a>
        </OverlayContainer>
      </Section>
    )}
  </>
)}
      </Container>
      
    </>
  );
};

export default PredictTissue;
