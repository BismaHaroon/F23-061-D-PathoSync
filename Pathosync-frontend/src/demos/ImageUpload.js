// ImageUploadPage.js
import React, { useState } from "react";
import tw from "twin.macro";
import { useNavigate } from "react-router-dom";
import LoadingDialog from "components/Loading/LoadingDialog";
import Header from "components/headers/light";
import Footer from "components/footers/FiveColumnWithBackground";
import OptionDialog from "components/AskAnnotationType/OptionDialoge";

const Container = tw.div`flex flex-col items-center justify-center h-screen p-8`;
const UploadWrapper = tw.div`mb-8 text-center space-x-2  text-2xl font-bold  text-purple-500  flex flex-col items-center`;
const ImageWrapper = tw.div`mb-8 flex flex-col items-center`;
const ImageBox = tw.div`border border-purple-500 rounded p-4 max-w-md w-full mx-auto`;
const ImagePreview = tw.img`max-w-full h-auto rounded shadow-lg`;
const Button = tw.button`mt-4 bg-primary-500 text-white px-6 py-3 rounded cursor-pointer`;
const AnnotationText = tw.p`mt-4 bg-primary-500 text-lg`;
const UnderlinedText = tw.span`text-purple-500 underline cursor-pointer`;
const InputImage =  tw.div`mb-8 text-center space-x-2 text-2xl font-bold text-purple-500 underline  justify-center `;

const ImageUploadPage = () => {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [isConfirmed, setIsConfirmed] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [showOptionDialog, setShowOptionDialog] = useState(false);
  const navigate = useNavigate();
  const [selectedOption, setSelectedOption] = useState("");

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    setUploadedImage(file);
  };

  const handleConfirm = async () => {
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append("image", uploadedImage);

      // Send a POST request to your Flask backend
      const response = await fetch("http://127.0.0.1:5000/upload", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        setIsConfirmed(true);

        // Fetch the processed image from the backend
        // Fetch the processed image from the backend using the same filename
        const processedResponse = await fetch(`http://127.0.0.1:5000/uploads/${uploadedImage.name.split('.')[0]}_normalized.png`);

        if (processedResponse.ok) {
          const processedImageBlob = await processedResponse.blob();
          setProcessedImage(URL.createObjectURL(processedImageBlob));
        }
      } else {
        console.error("Failed to upload image");
      }
    } catch (error) {
      console.error("Error during image upload:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleOptionSelect = (option) => {
    setSelectedOption(option);
    setShowOptionDialog(false);
    
  
    if (processedImage) {
      const encodedProcessedImage = encodeURIComponent(processedImage);
  
      // Determine the route based on the selected option
      let route = "";
      switch (option) {
        case "AnnotateCell":
          route = "/AnnotateCell";
          break;
        case "AnnotateTissue":
          route = `/AnnotateTissue/${encodedProcessedImage}`;
          break;
        case "AnnotateSAM":
          route = "/AnnotateSAM";
          break;
        default:
          console.error("Unknown annotation option:", option);
          return;
      }
  
      // Redirect to the determined route
      navigate(route);
    } else {
      console.error("Processed image URL is undefined.");
      // Handle the case where processedImage is undefined.
    }
  };

  return (
    <>
      <Header />
      <Container>
        <UploadWrapper>
          <label htmlFor="imageInput" css={tw`cursor-pointer justify-center hidden flex flex-col items-center `}>
            <div css={tw`mb-8 text-center space-x-2  text-purple-500  text-2xl font-bold flex flex-col items-center `}>
              <span>Click here to upload your image</span>
              <br/>
              </div>
              <input
  type="file"
  id="imageInput"
  accept="image/*"
  onChange={handleImageUpload}
  css={tw``}
  style={{
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    textAlign: 'center',
    marginTop: '1rem',
  }}
/>
              <br/>
           
          </label>
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
            <Button
              onClick={handleConfirm}
              disabled={!uploadedImage}
            >
              Confirm
            </Button>
          </ImageWrapper>
        )}
  
        {isConfirmed && (
          <ImageWrapper>
            <ImageBox>
              <p css={tw`text-lg font-bold`}>Processed Image:</p>
              <ImagePreview
                src={processedImage}
                alt="Processed Preview"
              />
            </ImageBox>
          </ImageWrapper>
        )}
  
        {isLoading && <LoadingDialog message="Preprocessing Image..." />}
  
        {isConfirmed && !showOptionDialog && (
          <AnnotationText>
            Image Confirmed.{' '}
            <UnderlinedText
              onClick={() => setShowOptionDialog(true)}
            >
              Select annotation type.
            </UnderlinedText>
          </AnnotationText>
        )}
  
        {showOptionDialog && (
          <OptionDialog onClose={() => setShowOptionDialog(false)} onOptionSelect={handleOptionSelect} />
        )}
      </Container>
      <Footer />
    </>
  );
};

export default ImageUploadPage;