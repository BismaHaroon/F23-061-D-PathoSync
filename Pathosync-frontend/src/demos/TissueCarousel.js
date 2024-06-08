import React from 'react';
import { FaAlignLeft } from 'react-icons/fa';
import Carousel from 'react-multi-carousel';
import 'react-multi-carousel/lib/styles.css';

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
    marginTop: '150px',
    marginBottom: '50px',
    width: '55%', 
    margin: '0 auto' ,
    marginLeft: '30px'
  };

  const carouselItemStyle = {
    position: 'relative',
    margin: '0 10px',
    cursor: 'pointer',
    border: '2px solid transparent',
  };

  const selectedStyle = {
    border: '5px solid #7946FD',
    borderRadius: '5px',
  };

  const carouselImageStyle = {
    width: '80%',
    height: 'auto',
    maxHeight: '150px',  // Adjust the max height to make images smaller
    objectFit: 'cover',
    borderRadius: '5px',
  };

const TissueCarousel = ({ images , onSelect}) => {
  return (
    <div style={carouselContainerStyle}>
    <Carousel responsive={responsive}>
      {images.map((image, index) => (
        <div key={index} style={carouselItemStyle}className="carousel-item" onClick={() => onSelect(image)}>  
          <img src={image.filepath} alt={image.filename} style={carouselImageStyle} />
        </div>
      ))}
    </Carousel>
    </div>




  );
};

export default TissueCarousel;
