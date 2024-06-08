import React from 'react';
import Carousel from 'react-multi-carousel';
import 'react-multi-carousel/lib/styles.css';

const SAMCarousel = ({ images, onSelect, selectedImages }) => {
  console.log("Images received in SAMCarousel:", images);

  const responsive = {
    superLargeDesktop: {
      breakpoint: { max: 4000, min: 3000 },
      items: 5
    },
    desktop: {
      breakpoint: { max: 3000, min: 1024 },
      items: 5
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
    maxHeight: '150px',
    objectFit: 'cover',
    borderRadius: '5px',
  };

  if (!images || images.length === 0) {
    return <div>No images available</div>;
  }

  return (
    <div style={carouselContainerStyle}>
      <Carousel responsive={responsive}>
        {images.map((image, index) => (
          <div
            key={index}
            style={selectedImages.includes(image) ? { ...carouselItemStyle, ...selectedStyle } : carouselItemStyle}
            onClick={() => onSelect(image)}
          >
            <img src={`http://localhost:5000/${image.filepath}`} alt={`Project Image ${index + 1}`} style={carouselImageStyle} />
          </div>
        ))}
      </Carousel>
    </div>
  );
};

export default SAMCarousel;
