import React, { useState } from 'react';
import tw from "twin.macro";

const Container = tw.div`flex flex-col items-center justify-center p-8`;
const Title = tw.h2`text-lg font-bold mb-4 text-purple-500`;
const FileInput = tw.input`mb-4`;
const LabelInput = tw.input`border border-gray-300 rounded p-2 mb-2`;
const Select = tw.select`border border-gray-300 rounded p-2 mb-4`;
const SubmitButton = tw.button`bg-purple-500 text-white px-4 py-2 rounded`;

function ModelTrain() {
  const [files, setFiles] = useState([]);
  const [labels, setLabels] = useState({});
  const [selectedModel, setSelectedModel] = useState('cnn');

  const handleFileChange = (event) => {
    setFiles([...event.target.files]);
    setLabels({});
  };

  const handleLabelChange = (file, label) => {
    setLabels({ ...labels, [file.name]: label });
  };

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  const handleSubmit = async () => {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
      formData.append('labels', labels[file.name] || '');
    });
    formData.append('model', selectedModel);

    // Replace with your backend endpoint
    const response = await fetch('http://127.0.0.1:5000/upload_train_data', {
      method: 'POST',
      body: formData
    });

    // Handle response
    console.log(await response.text())
  };

  return (
    <Container>
      <Title>Data Upload and Model Selection</Title>
      <FileInput type="file" multiple onChange={handleFileChange} />
      {files.map(file => (
        <div key={file.name}>
          <span>{file.name}</span>
          <LabelInput
            type="text"
            placeholder="Enter label (optional)"
            value={labels[file.name] || ''}
            onChange={(e) => handleLabelChange(file, e.target.value)}
          />
        </div>
      ))}
      <Select value={selectedModel} onChange={handleModelChange}>
        <option value="cnn">CNN</option>
        <option value="rcnn">RCNN</option>
        <option value="unet">U-Net</option>
      </Select>
      <SubmitButton onClick={handleSubmit}>Submit</SubmitButton>
    </Container>
  );
}

export default ModelTrain;
