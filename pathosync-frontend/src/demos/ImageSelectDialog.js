import React from 'react';
import tw, { styled } from "twin.macro";
import JSZip from 'jszip';
import { saveAs } from 'file-saver';

const DialogContainer = tw.div`fixed inset-0 bg-gray-500 bg-opacity-75 flex justify-center items-center z-50`;
const Dialog = tw.div`bg-white rounded-lg p-8 max-w-4xl w-full overflow-auto`;
const GridContainer = tw.div`grid grid-cols-4 gap-4`;
const Heading = tw.h2`text-xl font-semibold text-gray-700 mb-4`;
const InfoText = tw.p`text-sm text-gray-600 mb-4`;
const Button = tw.button`text-white bg-blue-500 hover:bg-blue-700 px-4 py-2 rounded transition-all ease-in-out duration-300`;
const Image = styled.img`
  ${tw`w-full h-auto rounded-lg shadow-md cursor-pointer transition-all duration-200 ease-in-out transform hover:scale-105`}
  object-fit: cover;
  height: 150px; // Define a fixed height for all images
  &:hover {
    ${tw`shadow-xl border border-gray-300`}
  }
`;


    // const downloadDataset = async (uploadedImages) => {
    //     const zip = new JSZip();
    //     const uploadsFolder = zip.folder("uploads");
    //     const masksFolder = zip.folder("masks");
      
    //     for (let i = 0; i < uploadedImages.length; i++) {
    //       const image = uploadedImages[i];
    //       uploadsFolder.file(image.name, image);
      
    //       // Adjust this URL/path as needed for your backend
    //       const maskedResponse = await fetch(`http://127.0.0.1:5000/display_nuclick_mask?image=${image.name}`);
    //       if (maskedResponse.ok) {
    //         const maskedBlob = await maskedResponse.blob();
    //         masksFolder.file(`masked_${image.name}`, maskedBlob);
    //       }
    //     }
      
    //     zip.generateAsync({ type: 'blob' }).then((content) => {
    //       saveAs(content, 'dataset.zip');
    //     });
    //   };
      const ImageSelectDialog = ({ images, onSelect, onClose, onDownloadDataset }) => {
    return (
        <DialogContainer onClick={onClose}>
          <Dialog onClick={(e) => e.stopPropagation()}>
            <Heading>Select One of the Images You Uploaded for Annotations</Heading>
            {/* <InfoText>To create a dataset for your model training, click the "Download Dataset" button. This will include two folders: one for your uploaded images and another for their masks.</InfoText>
            <Button onClick={() => onDownloadDataset()}>Download Dataset</Button> */}
            <GridContainer>
              {images.map((image, index) => (
                <Image
                  key={index}
                  src={URL.createObjectURL(image)}
                  alt={`Uploaded Preview ${index + 1}`}
                  onClick={() => onSelect(index)}
                />
              ))}
            </GridContainer>
          </Dialog>
        </DialogContainer>
      );
};

export default ImageSelectDialog;
