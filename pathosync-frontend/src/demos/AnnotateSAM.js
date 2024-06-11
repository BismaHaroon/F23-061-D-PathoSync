import Header from "components/headers/light";
import React, { useState, useRef, useEffect } from 'react';
import styled, { keyframes } from 'styled-components';
import tw from "twin.macro"; // Importing tw from twin.macro for utility-first CSS
import { useLocation } from "react-router-dom";
// Import Tailwind CSS styles
import 'tailwindcss/tailwind.css';
import SAMCarousel from "./SAMCarousel";
import { Link } from 'react-router-dom';
const CanvasContainer = tw.div`flex flex-col items-center justify-center w-full h-full mt-8`;
const ImagesDisplayContainer = tw.div`flex justify-center items-center w-full mt-4`;

// Animation for images entering the screen, adjusted to match AnnotateCell.js
const swoopInAnimation = keyframes`
  from {
    opacity: 0;
    transform: translateY(-1000px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
`;

const Canvas = styled.canvas`
  border: 4px solid #e2e8f0;
  background-color: #fff;
  transition: border-color 0.3s ease-in-out, width 0.5s ease-in-out, height 0.5s ease-in-out;
  &:hover {
    border-color: #cbd5e1;
  }
`;

const ImageContainer = styled.div`
  ${tw`m-2`}
  flex: 0 0 25%; // Makes each container take up exactly half of the parent container's width
  border: 3px solid #cbd5e1;
  animation: ${swoopInAnimation} 0.5s ease-out forwards;
  img {
    display: block;
    width: 100%; // Ensure the image fills the container
    max-height: 300px; // Adjust the maximum height as needed
    object-fit: contain; // Keeps the aspect ratio without cropping the image
  }
`;

const SegmentedImageWrapper = styled.div`
  ${tw`border border-gray-300 shadow-xl mt-4 p-4`}
  animation: ${swoopInAnimation} 0.5s ease-out forwards;
  img {
    display: block;
    width: auto;
    max-height: 400px;
  }
`;

const Button = styled.button`
  ${tw`px-4 py-2 text-sm rounded-md focus:outline-none focus:shadow-outline m-2`}
  background-color: #cbd5e1;
  &:hover {
    background-color: #b4c0d0;
  }
`;

const LabelInput = styled.input`
  ${tw`rounded-md p-2 mr-2`} // Using Tailwind classes for rounded corners, padding, and right margin
  height: 40px;
  width: 150px;
  margin-left: -55px;
  border: 2px solid #6415FF; // Thicker border and custom color
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); // Custom shadow

  ::placeholder { /* Chrome, Firefox, Opera, Safari 10.1+ */
    color: #6415FF;
    opacity: 0.5; // Ensure placeholder opacity is consistent
  }

  :-ms-input-placeholder { /* Internet Explorer 10-11 */
    color: #6415FF;
  }

  ::-ms-input-placeholder { /* Microsoft Edge */
    color: #6415FF;
  }
`;
const LabelInputContainer = tw.div`flex items-center mt-8`;
const ButtonContainer = styled.div`

  margin-top: ${({ hasImage }) => hasImage ? '20px' : '4px'}; /* Adjust space based on image presence */
`;
const DownloadContainer = styled.div`
  position: fixed;
  right: 20px;
  top: 100px; // Lowered as per requirement
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
const Heading = tw.h2`text-3xl font-bold text-gray-800 mb-4`;
const InfoText = tw.p`text-sm text-gray-600 mb-2`;
const App = () => {
  const canvasRef = useRef(null);
  const [drawing, setDrawing] = useState(false);
  const [startCoords, setStartCoords] = useState({ x: 0, y: 0 });
  const [endCoords, setEndCoords] = useState({ x: 0, y: 0 });
  const [boundingBoxes, setBoundingBoxes] = useState([]);
  const [image, setImage] = useState(null);
  const [currentLabel, setCurrentLabel] = useState('');
  const [segmentedImage, setSegmentedImage] = useState(null);
  const [maskedImage, setMaskedImage] = useState(null);
  const [downloadOption, setDownloadOption] = useState('');
  const [uploadedImage, setUploadedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [imageUrls, setImageUrls] = useState([]);
  const [selectedImages, setSelectedImages] = useState([]);
  const location = useLocation();
  const { images, project_name } = location.state || { images: [], project_name: '' };

// Log the received data
useEffect(() => {
  console.log('Received data in frontend:', { images, project_name });
}, [images, project_name]);

useEffect(() => {
  console.log('Project Name:', project_name);  // Print project ID to console
  const fetchImages = async () => {
    try {
      const urls = await Promise.all(
        images.map(async (image) => {
          const response = await fetch(`http://localhost:5000/${image.filepath}`);
          if (response.ok) {
            return image.filepath;
          } else {
            console.error(`Failed to fetch image: ${image.filepath}`);
            return null;
          }
        })
      );
      setImageUrls(urls.filter(url => url !== null));
    } catch (error) {
      console.error("Error fetching images:", error);
    } finally {
      setIsLoading(false);
    }
  };

  if (images.length > 0) {
    fetchImages();
  }
}, [ project_name]);

const handleImageSelect = async (image) => {
  setSelectedImages([image]);
  setUploadedImage(`http://localhost:5000/${image.filepath}`);

  try {
    const sanitizedProjectName = project_name.replace(/\s+/g, '_');
    const response = await fetch("http://localhost:5000/process-image-cell", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        project_name: sanitizedProjectName,
        image_path: image.filepath
      })
    });

    if (response.ok) {
      const data = await response.json();
      setUploadedImage(`http://localhost:5000/${data.image_path}`);
      setImage(`http://localhost:5000/${data.image_path}`);
      // Draw the selected image on the canvas
      const img = new Image();
      img.src = `http://localhost:5000/${data.image_path}`;
      img.onload = () => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
      };
    } else {
      console.error("Failed to process image");
    }
  } catch (error) {
    console.error("Error processing image:", error);
  }
};

  const handleDownloadImage = () => {
    let imageSrc = '';
    switch (downloadOption) {
      case 'orignal':
        imageSrc = image; // Assuming this is the URL to the annotated image
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


  const fetchMaskedImage = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/display_nuclick_mask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          image_path: uploadedImage // Pass the uploaded image path
        })
      });
      if (response.ok) {
        setMaskedImage(URL.createObjectURL(await response.blob()));
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
                image_path: uploadedImage  // Pass the uploaded image path
            }),
        });

        if (response.ok) {
            const data = await response.json();
            console.log('Received data from backend:', data);

            // Log the segmented image URL
            const segmentedImageUrl = `http://127.0.0.1:5000/uploads/${data.segmented_image_path}`;
            console.log('Segmented image URL:', segmentedImageUrl);

            // Set the segmented image URL based on the response
            setSegmentedImage(segmentedImageUrl);
        } else {
            console.error('Error sending bounding boxes:', response.status);
        }
    } catch (error) {
        console.error('Error sending bounding boxes:', error);
    }
};

useEffect(() => {
    if (segmentedImage) {
        console.log('Segmented image URL updated:', segmentedImage);
    }
}, [segmentedImage]);


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
      
      <h1 style={{ fontSize: '32px', fontWeight: 'bold', marginLeft: '50px' }}>SAM Annotation Tool</h1>
      
      <div style ={{marginLeft:'50px'}}>
      <InfoText>Draw & Send Bounding Boxes To Get Segmented Image. You May Also Add Labels.</InfoText>
      <Link 
  to="/CreateProject" 
  style={{ 
    display: 'inline-block',
    width: '60px', // Adjusted width
    padding: '5px',
    margin: '8px 0',
    marginLeft:'10px',
    marginBottom: '-20px',
    border: '1px solid #ccc', // Grey border
    borderRadius: '4px',
    marginTop: '10px',
    textDecoration: 'none', // Remove underline from the link
    backgroundColor: '#f0f0f0', // Button background color
    color: '#000', // Button text color
    textAlign: 'center' // Center text
  }}
>
  Back
</Link>
      </div>
      <CanvasContainer > 
        <div>

          {/* <LabelInputContainer style={{marginTop:'10px', marginBottom:'10px'}}> */}
          <label className="mr-2"></label>
          <LabelInput style={{height:'40px', width: '150px', marginLeft:'0px'}}
            type="text"
            value={currentLabel}
            onChange={handleLabelChange}
            placeholder="Enter Label Here"
          />
      <Button style={{backgroundColor: '#6415FF', color: '#ffffff', fontWeight: 'bold', width: '150px', marginLeft:'200px'}}
        onClick={fetchMaskedImage} // Add a click event handler to fetch and display the masked image
      > Fetch Mask
      </Button>
      
        <Button style={{backgroundColor: '#6415FF', color: '#ffffff', fontWeight: 'bold', width: '190px'}}
          onClick={sendBoundingBoxes}
        
        >Send Bounding Boxes
        </Button>
        
        
        
        </div>
        <Canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
        />
       
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
    <ImagesDisplayContainer>
      

    {segmentedImage && (
          <div><h1 style={{ fontSize: '22px', fontWeight: 'bold', textAlign: 'center' }}>Segmented Image</h1>
          <ImageContainer >
            
            <img src={segmentedImage} alt="Segmented Result" className="mt-2" />
          </ImageContainer></div>
        )}
        {maskedImage && (
          <div><h1 style={{ fontSize: '22px', fontWeight: 'bold', textAlign: 'center' }}>Masked Image</h1>
          <ImageContainer >
            
            <img src={maskedImage} alt="Segmented Result" className="mt-2" />
          </ImageContainer></div>
        )}
        
        </ImagesDisplayContainer>
        


      </CanvasContainer>
      <div><SAMCarousel images={imageUrls.map(filepath => ({ filepath }))} onSelect={handleImageSelect} selectedImages={selectedImages} /></div>
    </>
  );
};


export default App;