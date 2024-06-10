<<<<<<< HEAD
import React, { useState, useEffect, useRef } from "react";
import tw from "twin.macro";
import { useNavigate } from "react-router-dom";
import LoadingDialog from "components/Loading/LoadingDialog";
import Header from "components/headers/light";
import Footer from "components/footers/FiveColumnWithBackground";
import styled, { keyframes } from 'styled-components';

// Import Tailwind CSS styles
import 'tailwindcss/tailwind.css';

const CanvasContainer = tw.div`flex flex-col items-center justify-center w-full h-full mt-8`;
const ImagesDisplayContainer = tw.div`flex justify-center items-center items-start flex-wrap w-full mt-4`;
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

const Canvas = styled.canvas`
  border: 4px solid #e2e8f0;
  background-color: #fff;
  &:hover {
    border-color: #cbd5e1;
  }
  transition: width 0.5s ease-in-out, height 0.5s ease-in-out;
`;

const ImageContainer = styled.div`
  ${tw` m-2`}
  border: 3px solid #cbd5e1;
  animation: ${swoopInAnimation} 0.5s ease-out;
  img {
    display: block; /* Removes bottom space under image */
    width: auto;
    max-height: 600px;
  }
`;

const Button = styled.button`
  ${tw`px-4 py-2 text-sm rounded-md focus:outline-none focus:shadow-outline m-2`}
  background-color: #cbd5e1;
  &:hover {
    background-color: #b4c0d0;
  }
`;
const DownloadContainer = styled.div`
position: fixed;
right: 20px;
top: 50px;
display: flex;
flex-direction: column;
align-items: flex-start;
background-color: #f9fafb;
padding: 10px;
border-radius: 8px;
box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
transition: box-shadow 0.2s ease-in-out;

&:hover {
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
}

select, button {
  margin-top: 8px;
  background-color: #cbd5e1;
  border: 2px solid transparent;
  padding: 8px 12px;
  border-radius: 6px;
  transition: background-color 0.2s ease-in-out, border-color 0.2s ease-in-out;
  cursor: pointer;

  &:hover {
    background-color: #b4c0d0;
    border-color: #a4b6c2;
  }
}

button {
  display: flex;
  align-items: center;
  svg {
    margin-right: 5px;
  }
}
`;
const Label = tw.label`block text-sm font-medium text-gray-700`;
const InfoText = tw.p`text-sm text-gray-600 mb-2`;

const Heading = tw.h2`text-3xl font-bold text-gray-800 mb-4`;
const LabelInputContainer = tw.div`flex items-center justify-center mt-4`;
const LabelInput = styled.input`
  ${tw`border border-gray-300 rounded-md shadow-sm p-2 mr-2`}
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
  const [annotationEnabled, setAnnotationEnabled] = useState(false);
  const [maskedImage, setMaskedImage] = useState(null);
  const [downloadOption, setDownloadOption] = useState('');
  const handleDownloadImage = () => {
    let imageSrc = '';
    switch (downloadOption) {
      case 'annotated':
        imageSrc = imageUrl; // Assuming this is the URL to the annotated image
        break;
      case 'mask':
        imageSrc = maskedImage;
        break;
      case 'segmented':
        imageSrc = segmentedImage;
        break;
      default:
        alert('Please select an image type to download.');
        return;
    }

    // Trigger download
    const link = document.createElement('a');
    link.href = imageSrc;
    link.download = `${downloadOption}-image.png`; // Example filename, adjust as needed
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  const handleLabelChange = (e) => {
    setLabelInput(e.target.value);
  };

  const sendCoordinatesToBackend = async () => {
    try {
      setIsSending(true);

      // Prepare the data in the format expected by the backend.
      // Assuming the backend expects an array of objects with x and y properties.
      const formattedCoordinates = clickedCoordinates.map(coord => ({
        x: Math.round(coord.x),
        y: Math.round(coord.y),
      }));

      const response = await fetch("http://127.0.0.1:5000/segmentation", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formattedCoordinates),
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
      if (!annotationEnabled) return;
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      // Draw a dot on the canvas
      context.beginPath();
      context.arc(x, y, 3, 0, 2 * Math.PI);
      context.fillStyle = "green";
      context.fill();
      // context.fillText(labelInput, x + 5, y - 5);

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
  }, [imageUrl, annotationEnabled, labelInput]);

  const fetchMaskedImage = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/display_nuclick_mask");
      if (response.ok) {
        setMaskedImage(URL.createObjectURL(await response.blob()));
      } else {
        console.error("Failed to fetch masked image");
      }
    } catch (error) {
      console.error("Error fetching masked image:", error);
    }
  };

  return (
    <CanvasContainer>
      {/* <Heading>Cell Annotation Tool</Heading> */}
      {/* Other UI elements remain the same */}
      <InfoText>Click "Annotate" to start placing dots. Use "Segment Image" to process the image.</InfoText>
      <Button onClick={() => setAnnotationEnabled(!annotationEnabled)}>
        {annotationEnabled ? "Stop Annotation" : "Start Annotation"}
      </Button>
      <ImagesDisplayContainer>
        <Canvas ref={canvasRef} width="1000" height="600" />
        {segmentedImage && (
          <ImageContainer>
            <img src={segmentedImage} alt="Segmented" />
          </ImageContainer>
        )}
        {maskedImage && (
        <ImageContainer>
          <img src={maskedImage} alt="Masked" />
        </ImageContainer>
      )}
      </ImagesDisplayContainer>
      
      
      {annotationEnabled && (
        
  <LabelInputContainer>
    <label className="mr-2">Label: </label>
    <LabelInput
      type="text"
      value={labelInput}
      onChange={handleLabelChange}
      placeholder="Enter Label Here"
    />
    {/* Additional content related to the label input */}
  </LabelInputContainer>
)}
         <DownloadContainer>
      <select value={downloadOption} onChange={(e) => setDownloadOption(e.target.value)}>
        <option value="">Select Image to Download</option>
        <option value="annotated">Annotated Image</option>
        <option value="mask">Mask Image</option>
        <option value="segmented">Segmented Image</option>
      </select>

      <Button onClick={handleDownloadImage}>
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" className="bi bi-download" viewBox="0 0 16 16">
          <path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"/>
          <path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z"/>
        </svg>
        Download
      </Button>
    </DownloadContainer>
        <Button onClick={sendCoordinatesToBackend} disabled={isSending}>{isSending ? 'Loading...' : 'Segment Image'}</Button>
        <Button onClick={fetchMaskedImage}>Fetch Mask</Button>
     
    </CanvasContainer>
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
      
      <h1 style={{ fontSize: '32px', fontWeight: 'bold', marginLeft: '50px' }}>Cellular Annotation</h1>
      <div css={tw`flex h-screen`}>
        <div css={tw`flex-none w-full p-8`}>
          {isLoading ? (
            <LoadingDialog message="Fetching Uploaded Image..." />
          ) : null}
          <AnnotationCanvas imageUrl={uploadedImage} />
        </div>
      </div>
      
    </>
  );
};

export default AnnotateCell;
=======
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
>>>>>>> 8655ceccc37e8fd8d0bdcbd17d190dc036418d41
