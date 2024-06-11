import React from 'react';
import Carousel from 'react-multi-carousel';
import 'react-multi-carousel/lib/styles.css';

const WSICarousel = ({ images, onSelect, selectedImages }) => {
  console.log("Images received in WSICarousel:", images);

  const groupImagesById = (images) => {
    return images.reduce((groups, image) => {
      const id = image.filename;
      if (!groups[id]) {
        groups[id] = [];
      }
      groups[id].push(image);
      return groups;
    }, {});
  };

  const groupedImages = groupImagesById(images);

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

  return (
    <div>
      {Object.keys(groupedImages).map((id) => (
        <div key={id}>
          <h3>{id}</h3>
          <div style={carouselContainerStyle}>
            <Carousel responsive={responsive}>
              {groupedImages[id].map((image, index) => (
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
        </div>
      ))}
    </div>
  );
};

export default WSICarousel;
