import React, { useState } from 'react';
import tw from 'twin.macro';
import { useNavigate } from 'react-router-dom';
import LoadingDialog from 'components/Loading/LoadingDialog';
import Header from 'components/headers/light';
import Footer from 'components/footers/FiveColumnWithBackground';
import ErrorDialog from 'components/Error/ErrorDialog';
import WSIOptionDialog from 'components/AskAnnotationType/WSIOptionDialoge';
import PatchUploadDialog from './PatchUploadDialog';
import styled, { keyframes } from 'styled-components';
import Loading from 'components/Loading/LoadingDialog'; 
import 'tailwindcss/tailwind.css';


const UploadIcon = tw.span`inline-block text-xl text-gray-500`;
const Container = tw.div`flex flex-col items-center justify-center min-h-screen p-8 bg-gray-100`;
const UploadWrapper = tw.div`mb-8 text-center`;
const UploadArea = tw.div`border-2 border-dashed border-gray-300 rounded-lg p-8 bg-white shadow-lg hover:border-gray-400 cursor-pointer`;
const Button = styled.button`
  ${tw`mt-4 text-white px-8 py-2 rounded cursor-pointer transition duration-300 ease-in-out transform hover:scale-105 focus:outline-none`}
  background-color: #6b46c1; /* Tailwind Purple-600 */
  &:hover {
    background-color: #553c9a; /* Tailwind Purple-700 */
  }
`;
const Typography = tw.p`mb-4 text-lg text-gray-600`;
const swoopInAnimation = keyframes`
  from {
    opacity: 0;
    transform: translateX(50px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
`;
const PreviewImage = tw.img`max-w-full h-auto rounded shadow-lg`;
const PreviewWrapper = styled.div`
  ${tw`border border-purple-500 rounded p-4 max-w-md w-full mx-auto`}
  animation: ${swoopInAnimation} 0.5s ease-out;
`;

const WSIUploadPage = () => {
  const [uploadedWSI, setUploadedWSI] = useState(null);
  const [wsiPreview, setWsiPreview] = useState(null);
  const [isConfirmed, setIsConfirmed] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [showErrorDialog, setShowErrorDialog] = useState(false);
  const [showOptionDialog, setShowOptionDialog] = useState(false);
  const [showPatchUploadDialog, setShowPatchUploadDialog] = useState(false);
  const [selectedPatchForAnnotation, setSelectedPatchForAnnotation] = useState(null);
  const [showLoadingDialog, setShowLoadingDialog] = useState(false);
  const navigate = useNavigate();

  const handleWSIUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setUploadedWSI(file);
      setWsiPreview(URL.createObjectURL(file));
    }
  };

  const handlePatchUpload = (patchPreview) => {
    setSelectedPatchForAnnotation(patchPreview);
    setShowPatchUploadDialog(false);
    navigate('/AnnotateWSIPatch', { state: { patchImage: patchPreview } }); // Passing the patch image URL to the annotation page
  };
  const fetchThumbnail = async () => {
    const formData = new FormData();
    formData.append('wsi', uploadedWSI);

    setIsLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:5000/upload_wsi', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Upload failed');

      const thumbnailBlob = await response.blob();
      const thumbnailUrl = URL.createObjectURL(thumbnailBlob);
      setWsiPreview(thumbnailUrl);
      setIsLoading(false);
      setShowOptionDialog(true); // Show the option dialog after fetching thumbnail
    } catch (error) {
      setIsLoading(false);
      setShowErrorDialog(true);
    }
  };

  const handleConfirm = () => {
    // Fetch the thumbnail when confirming the WSI upload
    fetchThumbnail();
  };

  const handleOptionSelect = async (option) => {
    setShowOptionDialog(false);
    if (option === "Annotate WSI as a whole") {
      navigate('/AnnotateWholeWSI');
    } else if (option === "Annotate WSI by patches") {
      setShowLoadingDialog(true); 
      const formData = new FormData();
      formData.append('wsi', uploadedWSI);

      try {
        const response = await fetch('http://127.0.0.1:5000/create_patches', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) throw new Error('Patch creation failed');

        // Handle patch file download
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'patches.zip';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        // Redirect to patch upload screen
        setShowLoadingDialog(false);
        setShowPatchUploadDialog(true);
      } catch (error) {
        setShowLoadingDialog(false);
        setShowErrorDialog(true);
      }
    }
  };

  return (
    <>
      <Header />
      <Container>
      <Typography>Drag & Drop or Upload Whole Slide Image</Typography>
        <UploadWrapper>
          <UploadArea >
            <UploadIcon>📤</UploadIcon>
            <p>Drag and drop your image here, or click to select a file to upload.</p>
            <input
             
              type="file"
              id="imageInput"
              // accept="image/*"
              onChange={handleWSIUpload}
              multiple
              css={tw`hidden`}
            />
          </UploadArea>
        </UploadWrapper>

        {wsiPreview && !isConfirmed && (
          <PreviewWrapper>
            {/* <Typography>WSI Preview:</Typography> */}
            {/* <PreviewImage src={wsiPreview} alt="WSI Preview" /> */}
            <Button onClick={handleConfirm} disabled={!uploadedWSI}>
              Confirm
            </Button>
          </PreviewWrapper>
        )}

        {isLoading && <LoadingDialog message="Uploading WSI file..." />}
        {showErrorDialog && (
          <ErrorDialog
            message="Unsupported file format. Please upload a valid WSI file."
            onClose={() => setShowErrorDialog(false)}
          />
        )}
        {showOptionDialog && (
          <WSIOptionDialog
            onClose={() => setShowOptionDialog(false)}
            onOptionSelect={handleOptionSelect}
          />
        )}
        {showLoadingDialog && <LoadingDialog message="Creating patches..." />}
        <PatchUploadDialog
          isOpen={showPatchUploadDialog}
          onClose={() => setShowPatchUploadDialog(false)}
          onUpload={handlePatchUpload}
        />
      </Container>
      <Footer />
    </>
  );
};

export default WSIUploadPage;
