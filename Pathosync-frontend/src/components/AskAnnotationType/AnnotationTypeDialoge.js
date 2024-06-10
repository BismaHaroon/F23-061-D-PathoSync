import React from 'react';
import { useNavigate } from 'react-router-dom';
import tw from "twin.macro";

// Styled components
const Overlay = tw.div`fixed top-0 left-0 w-full h-full bg-gray-800 bg-opacity-75 z-50`;
const Dialog = tw.div`fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-white p-8 rounded-lg shadow-lg z-50`;
const Title = tw.p`text-lg font-bold mb-4 text-purple-500`;
const ButtonGroup = tw.div`w-full flex flex-col items-center`; // For side-by-side buttons
const ActionButtonGroup = tw.div`flex justify-center space-x-4`; // New container for action buttons
const CancelButtonContainer = tw.div`mt-4 flex justify-center`; // Container for the centered Cancel button
const Button = tw.button`bg-purple-500 text-white px-4 py-2 rounded hover:bg-purple-600 focus:outline-none focus:ring-2 focus:ring-purple-600 focus:ring-opacity-50`; // Adjusted
const CancelButton = tw(Button)`bg-gray-500 hover:bg-gray-600`; // Specific style for Cancel button

const AnnotationTypeDialoge = ({ onClose }) => {
  const navigate = useNavigate();

  const handleSelection = (path) => {
    navigate(path);
    onClose(); // Close the dialog after navigation
  };

  return (
    <>
      <Overlay onClick={onClose} />
      <Dialog>
        <Title>Select Annotation Type</Title>
        <ButtonGroup>
          <ActionButtonGroup>
            <Button onClick={() => handleSelection('/WSIUpload')}>Annotate WSI</Button>
            <Button onClick={() => handleSelection('/:option')}>Annotate Image</Button>
          </ActionButtonGroup>
        </ButtonGroup>
        <CancelButtonContainer>
          <CancelButton onClick={onClose}>Cancel</CancelButton>
        </CancelButtonContainer>
      </Dialog>
    </>
  );
};

export default AnnotationTypeDialoge;
