import Header from "components/headers/light";
import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';

// Import Tailwind CSS styles
import 'tailwindcss/tailwind.css';

// Create a styled component for the canvas
const Canvas = styled.canvas`
  border: 1px solid black;
`;

const AppWrapper = styled.div`
  max-width: 800px; /* Adjust as needed */
  margin: 0 auto;
`;

const LabelInput = styled.input`
  border: 1px solid #ced4da; /* Tailwind-like styling for border color */
  padding: 0.5rem; /* Tailwind-like styling for padding */
  margin-right: 0.5rem; /* Tailwind-like styling for margin */
`;

const SendButton = styled.button`
  background-color: #4caf50; /* Tailwind-like styling for background color */
  color: white; /* Tailwind-like styling for text color */
  padding: 0.5rem 1rem; /* Tailwind-like styling for padding */
  margin-top: 1rem; /* Tailwind-like styling for margin */
  cursor: pointer; /* Tailwind-like styling for cursor */
`;

const SegmentedImageWrapper = styled.div`
  border: 1px solid #ddd; /* Tailwind-like styling for border */
  padding: 1rem; /* Tailwind-like styling for padding */
  margin-top: 1rem; /* Tailwind-like styling for margin */
`;

const App = () => {
  const canvasRef = useRef(null);
  const [drawing, setDrawing] = useState(false);
  const [startCoords, setStartCoords] = useState({ x: 0, y: 0 });
  const [endCoords, setEndCoords] = useState({ x: 0, y: 0 });
  const [boundingBoxes, setBoundingBoxes] = useState([]);
  const [image, setImage] = useState(null);
  const [currentLabel, setCurrentLabel] = useState('');
  const [segmentedImage, setSegmentedImage] = useState(null);

  useEffect(() => {
    const fetchImage = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5000/get_latest_processed_image');
        if (response.ok) {
          const blob = await response.blob();
          setImage(URL.createObjectURL(blob));
        } else {
          console.error('Failed to fetch image');
        }
      } catch (error) {
        console.error('Error fetching image:', error);
      }
    };

    fetchImage();
  }, []);

  const fetchMaskedImage = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/display_mask");
      if (response.ok) {
        setSegmentedImage(URL.createObjectURL(await response.blob()));
      } else {
        console.error("Failed to fetch masked image");
      }
    } catch (error) {
      console.error("Error fetching masked image:", error);
    }
  };

  const drawImage = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (image) {
      const img = new Image();
      img.src = image;
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);

        boundingBoxes.forEach((box) => {
          ctx.strokeStyle = 'blue';
          ctx.lineWidth = 2;
          ctx.strokeRect(box.x, box.y, box.width, box.height);

          ctx.fillStyle = 'blue';
          ctx.font = '12px Arial';
          ctx.fillText(`Label: ${box.label}`, box.x, box.y - 5);
        });
      };
    }
  };

  const drawBoundingBox = (start, end) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    drawImage();
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.strokeRect(start.x, start.y, end.x - start.x, end.y - start.y);
  };

  const handleMouseDown = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setStartCoords({ x, y });
    setEndCoords({ x, y });
    setDrawing(true);
  };

  const handleMouseMove = (e) => {
    if (!drawing) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setEndCoords({ x, y });
    drawBoundingBox(startCoords, { x, y });
  };

  const handleMouseUp = () => {
    setDrawing(false);
    setBoundingBoxes([...boundingBoxes, { ...startCoords, width: endCoords.x - startCoords.x, height: endCoords.y - startCoords.y, label: currentLabel }]);
    setCurrentLabel('');
  };

  const handleLabelChange = (e) => {
    setCurrentLabel(e.target.value);
  };

  const sendBoundingBoxes = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/SAM', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          boundingBoxes,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        console.log(data);
        // Fetch and set the segmented image
        fetchSegmentedImage();
      } else {
        console.error('Error sending bounding boxes:', response.status);
      }
    } catch (error) {
      console.error('Error sending bounding boxes:', error);
    }
  };

  const fetchSegmentedImage = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/get_latest_SAM');
      if (response.ok) {
        const blob = await response.blob();
        setSegmentedImage(URL.createObjectURL(blob));
      } else {
        console.error('Failed to fetch segmented image');
      }
    } catch (error) {
      console.error('Error fetching segmented image:', error);
    }
  };

  useEffect(() => {
    drawImage();
  }, [image, boundingBoxes]);

  return (
    <>
      <Header />
      <AppWrapper className="p-4"> 
        <Canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
        />
        <div className="mt-4 flex items-center"> 
          <label className="mr-2">Label: </label>
          <LabelInput
            type="text"
            value={currentLabel}
            onChange={handleLabelChange}
            className="border p-1 w-32" /* Use Tailwind CSS classes for styling and set the width */

          />
        </div>
        <SendButton
          onClick={sendBoundingBoxes}
          className="bg-green-500 text-white p-2 mt-4 rounded" /* Use Tailwind CSS classes for styling */
        >
          Send Bounding Boxes
        </SendButton>
        <button
        onClick={fetchMaskedImage} // Add a click event handler to fetch and display the masked image
        style={{
          padding: '8px 16px',
          margin: '8px 45px',
          backgroundColor: '#6415FF',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer',
          width: '150px',
        }}
      >
        Fetch Mask
      </button>
        {segmentedImage && (
          <SegmentedImageWrapper className="mt-4">
            <h2>Segmented Result</h2>
            <img src={segmentedImage} alt="Segmented Result" className="mt-2" />
          </SegmentedImageWrapper>
        )}
      </AppWrapper>
    </>
  );
};


export default App;