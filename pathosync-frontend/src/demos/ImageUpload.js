// ImageUploadPage.js
import React, { useState } from "react";
import tw from "twin.macro";
import { useNavigate } from "react-router-dom";
import LoadingDialog from "components/Loading/LoadingDialog";
import Header from "components/headers/light";
import Footer from "components/footers/FiveColumnWithBackground";
import OptionDialog from "components/AskAnnotationType/OptionDialoge";
import ImageSelectDialog from './ImageSelectDialog';
import JSZip from 'jszip';
import { saveAs } from 'file-saver';

const Container = tw.div`flex flex-col items-center justify-center min-h-screen p-8 bg-gray-100`;
const UploadWrapper = tw.div`mb-8 text-center`;
const UploadArea = tw.div`border-2 border-dashed border-gray-300 rounded-lg p-8 bg-white shadow-lg hover:border-gray-400 cursor-pointer`;
const UploadIcon = tw.span`inline-block text-xl text-gray-500`;
const Button = tw.button`mt-4 text-white px-8 py-2 rounded cursor-pointer transition duration-300 ease-in-out transform hover:scale-105 focus:outline-none bg-purple-500 hover:bg-purple-600`;
const Typography = tw.p`mb-4 text-lg text-gray-600`;
const ImageWrapper = tw.div`mb-8 flex flex-col items-center`;
const ImageBox = tw.div`border border-purple-500 rounded p-4 max-w-md w-full mx-auto`;
const ImagePreview = tw.img`max-w-full h-auto rounded shadow-lg`;

const ImageUploadPage = () => {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [isConfirmed, setIsConfirmed] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [showOptionDialog, setShowOptionDialog] = useState(false);
  const navigate = useNavigate();
  const [selectedOption, setSelectedOption] = useState("");
  const [uploadedImages, setUploadedImages] = useState([]); // Adjust to handle multiple images
  const [selectedImage, setSelectedImage] = useState(null); // Store the selected image file
  const [showImageSelectDialog, setShowImageSelectDialog] = useState(false);

  const handleImageUpload = (event) => {
    const files = Array.from(event.target.files);
    setUploadedImages(files);
    setShowImageSelectDialog(true);
  };
  const handleImageSelection = (index) => {
    const selectedFile = uploadedImages[index];
    setSelectedImage(selectedFile);
    setUploadedImage(selectedFile);
    setShowImageSelectDialog(false); // Close the dialog after selection
    // Proceed with the confirmation and processing for the selectedFile
  };

  const handleConfirm = async () => {
    if (!selectedImage) return;
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append("image", selectedImage);

      // Send a POST request to your Flask backend
      const response = await fetch("http://127.0.0.1:5000/upload", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        setIsConfirmed(true);

        // Fetch the processed image from the backend
        // Fetch the processed image from the backend using the same filename
        const processedResponse = await fetch(`http://127.0.0.1:5000/uploads/${selectedImage.name.split('.')[0]}_normalized.png`);

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
  const downloadDataset = async (uploadedImages) => {
    const zip = new JSZip();
    const uploadsFolder = zip.folder("uploads");
    const masksFolder = zip.folder("masks");
  
    for (let i = 0; i < uploadedImages.length; i++) {
      const image = uploadedImages[i];
      uploadsFolder.file(image.name, image);
  
      const maskedResponse = await fetch(`http://127.0.0.1:5000/display_nuclick_mask?image=${image.name}`);
      if (maskedResponse.ok) {
        const maskedBlob = await maskedResponse.blob();
        masksFolder.file(`masked_${image.name}`, maskedBlob);
      }
    }
  
    zip.generateAsync({ type: 'blob' }).then((content) => {
      saveAs(content, 'dataset.zip');
    });
  };
  return (
    <>
      <Header />
      <Container>
        <Typography>Drag & Drop or Upload Image</Typography>
        <UploadWrapper>
          <UploadArea >
            <UploadIcon>📤</UploadIcon>
            <p>Drag and drop your image here, or click to select a file to upload.</p>
            <input
             
              type="file"
              id="imageInput"
              accept="image/*"
              onChange={handleImageUpload}
              multiple
              css={tw`hidden`}
            />
          </UploadArea>
        </UploadWrapper>
        {showImageSelectDialog && (
        <ImageSelectDialog
          images={uploadedImages}
          onSelect={handleImageSelection}
          onClose={() => setShowImageSelectDialog(false)}
          onDownloadDataset={() => downloadDataset(uploadedImages)}
          
        />
      )}
        {selectedImage && !isConfirmed && (
          <ImageWrapper>
            <ImageBox>
              <Typography>Uploaded Image:</Typography>
              <ImagePreview
                src={URL.createObjectURL(selectedImage)}
                alt="Uploaded Preview"
              />
            </ImageBox>
            <Button onClick={handleConfirm} disabled={!uploadedImage}>
              Confirm
            </Button>
          </ImageWrapper>
        )}

        {isConfirmed && processedImage && (
          <ImageWrapper>
            <ImageBox>
              <Typography>Processed Image:</Typography>
              <ImagePreview src={processedImage} alt="Processed Preview" />
            </ImageBox>
          </ImageWrapper>
        )}

        {isLoading && <LoadingDialog message="Preprocessing Image..." />}

        {isConfirmed && !showOptionDialog && (
          <div css={tw`mt-4`}>
            <Button onClick={() => setShowOptionDialog(true)}>Select Annotation Type</Button>
          </div>
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