import React, { useState } from 'react';
import Modal from 'react-modal';
import tw, { styled } from 'twin.macro';

Modal.setAppElement('#root');

const StyledModal = styled(Modal)`
  ${tw`fixed inset-0 flex items-center justify-center`}
  outline: none;
  background-color: rgba(0, 0, 0, 0.5); // Semi-transparent backdrop
`;

const ModalContent = styled.div`
  ${tw`w-full max-w-md p-6 bg-white rounded-lg shadow-xl`}
  position: relative;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08); // Highlighting boundary with shadow
`;

const Title = styled.h2`
  ${tw`text-2xl font-bold text-gray-900 mb-4`}
`;

const Description = styled.p`
  ${tw`text-gray-700 mb-4`}
`;

const FileInputButton = styled.label`
  ${tw`block w-full px-4 py-2 mt-2 text-sm font-semibold text-gray-800 bg-white border border-gray-300 rounded-lg cursor-pointer hover:bg-gray-500 focus:outline-none`}
  display: inline-flex;
  align-items: center;
  justify-content: center;
`;

const FileInput = styled.input`
  ${tw`hidden`}
`;

const PreviewImage = styled.img`
  ${tw`w-full mt-4 rounded-md`}
  max-height: 200px; // Limit the preview image height
  object-fit: cover;
`;

const SubmitButton = styled.button`
  ${tw`px-4 py-2 mt-4 text-sm text-white bg-purple-600 rounded-md hover:bg-blue-700 focus:outline-none`}
`;

const PatchUploadDialog = ({ isOpen, onClose, onUpload }) => {
  const [selectedPatch, setSelectedPatch] = useState(null);
  const [patchPreview, setPatchPreview] = useState(null);

  const handlePatchUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedPatch(file);
      setPatchPreview(URL.createObjectURL(file));
    }
  };

  const handleSubmit = () => {
    if (selectedPatch) {
      onUpload(patchPreview); // Pass the patch preview URL to the parent component
    }
  };

  return (
    <StyledModal isOpen={isOpen} onRequestClose={onClose} contentLabel="Upload Patch Dialog">
      <ModalContent>
        <Title>Upload a Patch for Annotation</Title>
        <Description>Please upload a patch to proceed with annotation.</Description>
        <FileInputButton>
          Select Patch
          <FileInput type="file" onChange={handlePatchUpload} />
        </FileInputButton>
        {patchPreview && <PreviewImage src={patchPreview} alt="Patch Preview" />}
        <SubmitButton onClick={handleSubmit}>Upload Patch</SubmitButton>
      </ModalContent>
    </StyledModal>
  );
};

export default PatchUploadDialog;
