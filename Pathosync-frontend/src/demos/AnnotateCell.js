import React, { useState, useEffect } from "react";
import tw from "twin.macro";
import { useNavigate } from "react-router-dom";
import LoadingDialog from "components/Loading/LoadingDialog";
import Header from "components/headers/light";
import Footer from "components/footers/FiveColumnWithBackground";
import styled from 'styled-components';

// Import Tailwind CSS styles
import 'tailwindcss/tailwind.css';

const LabelInput = styled.input`
  border: 1px solid #ced4da; /* Tailwind-like styling for border color */
  padding: 0.5rem; /* Tailwind-like styling for padding */
  margin-right: 0.5rem; /* Tailwind-like styling for margin */
`;

const AnnotationCanvas = ({ imageUrl }) => {
  const [Cx, setCx] = useState([]); // Array for x-coordinates
  const [Cy, setCy] = useState([]); // Array for y-coordinates
  const [clickedCoordinates, setClickedCoordinates] = useState([]); // Clicked coordinates
  const [hoverCoordinates, setHoverCoordinates] = useState(null); // Hover coordinates
  const [isSending, setIsSending] = useState(false); 
  const canvasRef = React.useRef(null);
  const [currentLabel, setCurrentLabel] = useState('');
  const [labelInput, setLabelInput] = useState(""); // User-entered label
  const [clickedAnnotations, setClickedAnnotations] = useState([]); // Clicked annotations with labels
  const [segmentedImage, setSegmentedImage] = useState(null); 

  const fetchMaskedImage = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/display_nuclick_mask");
      if (response.ok) {
        setSegmentedImage(URL.createObjectURL(await response.blob()));
      } else {
        console.error("Failed to fetch masked image");
      }
    } catch (error) {
      console.error("Error fetching masked image:", error);
    }
  };

  const handleLabelChange = (e) => {
    setCurrentLabel(e.target.value);
  };
  
  const sendCoordinatesToBackend = async () => {
    try {
      setIsSending(true);
      const response = await fetch("http://127.0.0.1:5000/segmentation", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(clickedCoordinates),
      });

      if (response.ok) {
        console.log("Coordinates sent to backend successfully");

        // Fetch and set the segmented image URL
        const segmentedResponse = await fetch("http://127.0.0.1:5000/nuclick_result");
        if (segmentedResponse.ok) {
          setSegmentedImage(URL.createObjectURL(await segmentedResponse.blob()));
        } else {
          console.error("Failed to fetch segmented image");
        }

        // Optionally, you can reset the clickedCoordinates array after successful submission
        setClickedCoordinates([]);
      } else {
        console.error("Failed to send coordinates to backend");
      }
    } catch (error) {
      console.error("Error sending coordinates to backend:", error);
    } finally {
      setIsSending(false);
    }
  };
  useEffect(() => {

    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    const image = new Image();

    const handleCanvasClick = async(event) => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      // Draw a dot on the canvas
      context.beginPath();
      context.arc(x, y, 3, 0, 2 * Math.PI);
      context.fillStyle = "green";
      context.fill();

      // Save the click position
      setCx((prevCx) => [...prevCx, x]);
      setCy((prevCy) => [...prevCy, y]);
      setClickedCoordinates((prevCoordinates) => [...prevCoordinates, { x, y }]);
      setClickedAnnotations((prevAnnotations) => [
        ...prevAnnotations,
        { x, y, label: labelInput },
      ]);
    };

    const handleCanvasHover = (event) => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      setHoverCoordinates({ x, y });
    };

    const handleCanvasLeave = () => {
      setHoverCoordinates(null);
    };

    image.onload = () => {
      // Set canvas dimensions to match the image dimensions
      canvas.width = image.width;
      canvas.height = image.height;

      context.drawImage(image, 0, 0, canvas.width, canvas.height);
    };

    if (imageUrl) {
      image.src = imageUrl;
    }

    canvas.addEventListener("click", handleCanvasClick);
    canvas.addEventListener("mousemove", handleCanvasHover);
    canvas.addEventListener("mouseleave", handleCanvasLeave);

    return () => {
      canvas.removeEventListener("click", handleCanvasClick);
      canvas.removeEventListener("mousemove", handleCanvasHover);
      canvas.removeEventListener("mouseleave", handleCanvasLeave);
    };
  }, [imageUrl]);


  return (
    <div css={tw`flex items-center justify-center h-full`}>
      <div css={tw`flex flex-col items-center`} style={{ width: '100%', maxWidth: '800px' }}>
        <div css={tw`mb-4`} style={{ marginLeft: '50px', marginTop: '20px', width: '100%', maxWidth: '340px', marginBottom: '20px' }}>
          <canvas
            ref={canvasRef}
            css={tw`mb-4`}
            style={{ border: '1px solid black', maxWidth: '100%' }}
          />
        </div>
        <div className="mt-4 flex items-center"> 
          <label className="mr-2">Label: </label>
          <LabelInput
            type="text"
            value={currentLabel}
            onChange={handleLabelChange}
            className="border p-1 w-32" /* Use Tailwind CSS classes for styling and set the width */

          />
        </div>
        <div
          css={tw`flex items-center`}
          style={{ width: '100%', maxWidth: '800px', marginLeft: '50px' }}
        >
          <div css={tw`mr-4`} style={{ width: '50%' }}>
            {segmentedImage && (
              <div css={tw`mb-4`} style={{ width: '100%', position: 'absolute' }}>
                <img
                  src={segmentedImage}
                  alt="Segmented"
                  style={{ maxWidth: '100%', position: 'absolute', top: '-277px', left: '300px' }}
                />
              </div>
            )}
          </div>
          {/* Button using old formatting */}
          <button
            onClick={sendCoordinatesToBackend}
            disabled={isSending}
            style={{
              padding: '8px 16px',
              margin: '8px 45px',
              backgroundColor: '#6415FF',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              width: '150px', // Adjusted width
            }}
          >
            {isSending ? 'Loading...' : 'Segment Image'}
          </button>
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
        </div>
      </div>
    </div>
  );
};

const AnnotateCell = () => {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchUploadedImage = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5000/get_latest_nuclick");
        if (response.ok) {
          setUploadedImage(URL.createObjectURL(await response.blob()));
        } else {
          console.error("Failed to fetch uploaded image");
        }
      } catch (error) {
        console.error("Error fetching uploaded image:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchUploadedImage();
  }, []);

  return (
    <>
      <Header />
      <h1 style={{ fontSize: '32px', fontWeight: 'bold', marginLeft: '50px' }}>Cellular Annotation</h1> {/* Bigger and bold heading */}
      <div css={tw`flex h-screen`}>
        <div css={tw`flex-none w-full p-8`}>
          {isLoading ? (
            <LoadingDialog message="Fetching Uploaded Image..." />
          ) : null}
          <AnnotationCanvas imageUrl={uploadedImage} />
        </div>
      </div>
      <Footer css={tw`mt-auto`} />
    </>
  );
};

export default AnnotateCell;