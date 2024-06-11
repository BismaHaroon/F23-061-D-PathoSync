import React, { useState, useEffect,useRef } from "react";
import tw from "twin.macro";
import { useLocation } from "react-router-dom";
import LoadingDialog from "components/Loading/LoadingDialog";
import Header from "components/headers/light";
import Footer from "components/footers/FiveColumnWithBackground";
import styled, { keyframes } from 'styled-components';
import CustomCarousel from "./CustomCarousel";
import { Link } from 'react-router-dom';
// Import Tailwind CSS styles
import 'tailwindcss/tailwind.css';

const CanvasContainer = tw.div`flex flex-col items-center justify-center w-full h-full mt-8`;
const ImagesDisplayContainer = tw.div`flex flex-wrap justify-center items-start w-full mt-4`;
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
${tw`m-2`}
flex: 1 1 auto;  // Adjust this to control how much space each container should take
max-width: 25%;  // Prevents the image from taking more than 45% of the parent's width
border: 3px solid #cbd5e1;
animation: ${swoopInAnimation} 0.5s ease-out;
width: auto;

img {
  display: block; // Ensures the image is a block to prevent extra space at the bottom
  width: 150%;  // Ensures the image takes the full width of the container
  height: auto; // Keeps the image's aspect ratio
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


const AnnotationCanvas = ({ imageUrl, uploadedImage }) => {
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
  const [originalDimensions, setOriginalDimensions] = useState({ width: 0, height: 0 });

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

  const handleLabelChange = (e) => {
    setLabelInput(e.target.value);
  };
  
  const sendCoordinatesToBackend = async () => {
    setIsSending(true);
    const formattedCoordinates = clickedCoordinates.map(coord => ({
      x: Math.round(coord.x), // These are already adjusted to original scale
      y: Math.round(coord.y),
    }));
  
    try {
      const response = await fetch("http://127.0.0.1:5000/segmentation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          coordinates: formattedCoordinates,
          image_path: uploadedImage // Pass the path of the processed image
        }),
      });
  
      if (response.ok) {
        const data = await response.json();
        setSegmentedImage(`http://127.0.0.1:5000/${data.segmented_image_path}`);
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
  const rect = canvasRef.current.getBoundingClientRect();
  const scaleX = originalDimensions.width / rect.width;
  const scaleY = originalDimensions.height / rect.height;

  const x = (event.clientX - rect.left) * scaleX;
  const y = (event.clientY - rect.top) * scaleY;
    
      // Draw a dot at the clicked position (scaled down for display)
      const context = canvas.getContext("2d");
      context.beginPath();
      context.arc(event.clientX - rect.left, event.clientY - rect.top, 3, 0, 2 * Math.PI);
      context.fillStyle = "green";
      context.fill();
    
      // Save the click position in the original image scale
      setCx(prevCx => [...prevCx, x]);
      setCy(prevCy => [...prevCy, y]);
      setClickedCoordinates(prevCoordinates => [...prevCoordinates, { x, y }]);
      setClickedAnnotations(prevAnnotations => [
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
      // Set the original image dimensions
      setOriginalDimensions({ width: image.naturalWidth, height: image.naturalHeight });
  
      // OG Image scaling
      const scale = 1.5; // Set scale as needed
      const canvas = canvasRef.current;
      canvas.width = image.naturalWidth * scale;
      canvas.height = image.naturalHeight * scale;
  
      const ctx = canvas.getContext('2d');
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
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

  return (
    
    <CanvasContainer>
        
      <div>
      <div>
      <Button style={{marginLeft:'20px', backgroundColor: '#6415FF', color: '#ffffff', fontWeight: 'bold', width: '150px'}} onClick={() => setAnnotationEnabled(!annotationEnabled)}>
        {annotationEnabled ? "Stop Annotation" : "Start Annotation"}
      </Button>
      
      <Button style={{marginLeft:'60px', backgroundColor: '#6415FF', color: '#ffffff', fontWeight: 'bold', width: '140px'}} onClick={sendCoordinatesToBackend} disabled={isSending}>{isSending ? 'Loading...' : 'Segment Image'}</Button>
      
      </div>
        <Button style={{marginLeft: '240px',backgroundColor: '#6415FF', color: '#ffffff', fontWeight: 'bold', width: '140px'}} onClick={fetchMaskedImage}>Fetch Mask</Button>
        
        
        <LabelInputContainer style={{marginTop:'-44px', marginLeft: '-330px'}}>
          <label className="mr-2"></label>
          <LabelInput style={{height:'40px', width: '150px', marginLeft:'140px'}}
            type="text"
            value={labelInput}
            onChange={handleLabelChange}
            placeholder="Enter Label Here"
          />
          {/* Additional content related to the label input */}
        </LabelInputContainer>
      
        </div>
      <ImagesDisplayContainer  style={{marginBottom: '20px'}}>
        <Canvas ref={canvasRef} width="1000" height="600" />
        {segmentedImage && (
          <ImageContainer >
            <img src={segmentedImage} alt="Segmented" />
          </ImageContainer>
        )}

        
        {maskedImage && (
        <ImageContainer>
          <img src={maskedImage} alt="Masked" />
        </ImageContainer>
      )}
      </ImagesDisplayContainer>
      
      
      
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
        {/* <Button onClick={sendCoordinatesToBackend} disabled={isSending}>{isSending ? 'Loading...' : 'Segment Image'}</Button>
        <Button onClick={fetchMaskedImage}>Fetch Mask</Button> */}
     
    </CanvasContainer>
  );
};

const AnnotateCell = () => {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [imageUrls, setImageUrls] = useState([]);
  const [selectedImages, setSelectedImages] = useState([]);
  const location = useLocation();
  const { images,  project_name } = location.state || { images: [],  project_name: '' };

  // Debugging: Log the received data
  console.log('Received data in frontend:', { images,  project_name });

  useEffect(() => {
    console.log('Project Name:', project_name);  // Print project ID to console
    const fetchImages = async () => {
      try {
        const urls = await Promise.all(
          images.map(async (image) => {
            const response = await fetch(`http://localhost:5000/${image.filepath}`);
            if (response.ok) {
              return URL.createObjectURL(await response.blob());
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
      } else {
        console.error("Failed to process image");
      }
    } catch (error) {
      console.error("Error processing image:", error);
    }
};

  
  return (
    <>
      <Header />
    
      <h1 style={{ fontSize: '32px', fontWeight: 'bold', marginLeft: '50px' }}>Cellular Annotation</h1>
      
      <div style ={{marginLeft:'50px'}}>
      <InfoText>Click "Start Annotation" to start placing dots.</InfoText>
      <InfoText style={{marginBottom:'10px'}}>Use "Segment Image" to process the image.</InfoText>
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
      
      <div css={tw`flex h-screen`}>
        <div css={tw`flex-none w-full p-8`}>
          {isLoading ? (
            <LoadingDialog message="Fetching Uploaded Image..." />
          ) : null}
          <CustomCarousel images={images} onSelect={handleImageSelect} selectedImages={selectedImages} />

          <AnnotationCanvas imageUrl={uploadedImage} uploadedImage={uploadedImage} />

          
        </div>
        
        
      </div>
      <Footer css={tw`mt-auto`} />
    </>
  );
};

export default AnnotateCell;