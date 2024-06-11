// ClassImagesCarousel.js
import React from 'react';
import Carousel from 'react-multi-carousel';
import 'react-multi-carousel/lib/styles.css';

const responsive = {
  superLargeDesktop: {
    breakpoint: { max: 4000, min: 3000 },
    items: 3
  },
  desktop: {
    breakpoint: { max: 3000, min: 1024 },
    items: 3
  },
  tablet: {
    breakpoint: { max: 1024, min: 464 },
    items: 2
  },
  mobile: {
    breakpoint: { max: 464, min: 0 },
    items: 1
  }
};
const carouselContainerStyle = {
  marginTop: '20px',
  marginBottom: '20px',
  width: '60%',
};

const carouselItemStyle = {
  position: 'relative',
  margin: '0 20px',
  cursor: 'pointer',
  border: '2px solid transparent',
};

const selectedStyle = {
  border: '5px solid #7946FD',
  borderRadius: '5px',
};

const carouselImageStyle = {
  width: '100%',
  height: 'auto',
  maxHeight: '150px',  // Adjust the max height to make images smaller
  objectFit: 'cover',
  borderRadius: '5px',
};

const AnnotatedImagesDisplay = ({ images }) => {
  return (
    <div style={carouselContainerStyle}>
    <Carousel responsive={responsive}>
      {images.map((url, idx) => (
        <div key={idx} style={{ padding: '10px' }}>
          <img
            src={url}
            alt={`Image ${idx + 1}`}
            style={carouselImageStyle}
          />
        </div>
      ))}
    </Carousel>
    </div>
  );
};

export default AnnotatedImagesDisplay;
