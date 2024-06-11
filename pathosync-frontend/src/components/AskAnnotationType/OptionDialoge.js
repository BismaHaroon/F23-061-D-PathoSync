// OptionDialog.js
import React from "react";
import tw from "twin.macro";

const Overlay = tw.div`fixed top-0 left-0 w-full h-full bg-gray-800 bg-opacity-75 z-50`;
const Dialog = tw.div`fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-white p-8 rounded z-50`;
const Title = tw.p`text-lg font-bold mb-4 text-purple-500`;
const OptionButtons = tw.div`flex items-center justify-center space-x-4`;
const OptionButton = tw.button`bg-purple-500 text-white px-4 py-2 rounded`;

const OptionDialog = ({ onClose, onOptionSelect }) => {
  const handleOptionClick = (option) => {
    onOptionSelect(option);
    onClose();
  };

  return (
    <>
      <Overlay onClick={onClose} />
      <Dialog>
        <Title>Select an option:</Title>
        <OptionButtons>
          <OptionButton onClick={() => handleOptionClick("AnnotateCell")}>
            Cell Annotation
          </OptionButton>
          <OptionButton onClick={() => handleOptionClick("AnnotateTissue")}>
            Tissue Annotation
          </OptionButton>
          <OptionButton onClick={() => handleOptionClick("AnnotateSAM")}>
            SAM Annotation
          </OptionButton>
        </OptionButtons>
      </Dialog>
    </>
  );
};

export default OptionDialog;
