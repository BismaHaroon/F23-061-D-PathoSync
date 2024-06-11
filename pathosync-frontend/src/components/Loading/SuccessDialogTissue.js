import React from "react";
import tw from "twin.macro";
import { useNavigate } from "react-router-dom";

// Styled components from the given example
const Overlay = tw.div`fixed top-0 left-0 w-full h-full bg-gray-800 bg-opacity-75 z-50`;
const Dialog = tw.div`fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-white p-8 rounded-lg shadow-lg z-50`;
const Title = tw.p`text-lg font-bold mb-4 text-purple-500`;
const ActionButtonGroup = tw.div`flex justify-center space-x-4`; // Container for action buttons
const CancelButtonContainer = tw.div`mt-4 flex justify-center`; // Container for the centered Cancel button
const Button = tw.button`bg-purple-500 text-white px-4 py-2 rounded hover:bg-purple-600 focus:outline-none focus:ring-2 focus:ring-purple-600 focus:ring-opacity-50`;
const CancelButton = tw(Button)`bg-gray-500 hover:bg-gray-600`; // Specific style for Cancel button

const SuccessDialogTissue = ({ message, onDone, onProceed, onRetrain, accuracy, loss }) => {
  const navigate = useNavigate();

  const handleProceed = () => {
    onProceed(); // Call the onProceed callback passed from the parent component
    navigate("/Predictions"); // Navigate to /PredictTissue
  };

  return (
    <>
      <Overlay />
      <Dialog>
        <Title>{message}</Title>
        
          <p>
            Accuracy: {accuracy}, Loss: {loss}
          </p>
       
        <ActionButtonGroup>
          <Button onClick={onDone}>Done</Button>
          <Button onClick={handleProceed}>Proceed</Button>
          <Button onClick={onRetrain}>Retrain</Button>
        </ActionButtonGroup>
      </Dialog>
    </>
  );
};

export default SuccessDialogTissue;
