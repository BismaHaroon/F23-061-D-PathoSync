import React, { useState } from "react";
import styled from "styled-components";
import tw from "twin.macro";
import { SectionHeading, Subheading as SubheadingBase } from "components/misc/Headings.js";
import { SectionDescription } from "components/misc/Typography.js";
import AnnotationTypeDialoge from "components/AskAnnotationType/AnnotationTypeDialoge"; // Make sure the import path is correct
import defaultCardImage from "images/shield-icon.svg";
import { Link } from "react-router-dom";
import { ReactComponent as SvgDecoratorBlob3 } from "images/svg-decorator-blob-3.svg";

import BloodCells from "images/blood_cells.svg";
import Tissue from "images/tissues.png";
import ModelTraining from "images/brain.png";
import Preprocess from "images/image-processing.png";

const Container = tw.div`relative`;

const TwoColumnContainer = styled.div`
  ${tw`flex flex-col items-center md:items-stretch md:flex-row flex-wrap md:justify-center max-w-screen-lg mx-auto py-20 md:py-24`}
`;

const Subheading = tw(SubheadingBase)`mb-4`;
const Heading = tw(SectionHeading)`w-full`;
const Description = tw(SectionDescription)`w-full text-center`;

const VerticalSpacer = tw.div`mt-10 w-full`;

const Column = styled.div`
  ${tw`lg:w-1/2 max-w-sm`}
`;

const Card = styled.div`
  ${tw`flex flex-col items-center text-center h-full mx-4 px-2 py-8 cursor-pointer transition-all duration-300 ease-in-out`}

  .imageContainer {
    ${tw`border rounded-full p-5 flex-shrink-0`}
    img {
      ${tw`w-12 h-12 transition-transform duration-300 ease-in-out`}
      transform: translateY(0); /* Starting state */
      &:hover {
        transform: translateY(-5px); /* Moves the image up slightly on hover */
      }
    }
  }

  .textContainer {
    ${tw`mt-4`}
  }

  .title {
    ${tw`tracking-wide font-bold text-2xl leading-none cursor-pointer transition-colors duration-300 ease-in-out`}
    color: #7c3aed ; /* Initial color */
    &:hover {
      color:
      #4c51bf; /* Lightens the color on hover */
    }
  }
  
  
  &:hover .imageContainer img {
    transform: translateY(-5px); /* Connects the hover effect of the title with the image */
  }
`;

const DecoratorBlob = styled(SvgDecoratorBlob3)`
  ${tw`pointer-events-none absolute right-0 bottom-0 w-64 opacity-25 transform translate-x-32 translate-y-48 `}
`;

export default ({
    cards = null,
    heading = "Services We offer",
    
    description="Explore our intuitive and user friendly histopathology tools"
  }) => {
    const [isDialogOpen, setDialogOpen] = useState(false);

    // Function to show dialog
    const showDialog = () => setDialogOpen(true);
    // Function to hide dialog
    const hideDialog = () => setDialogOpen(false);
    
    const handleCardClick = (card) => {
        if (card.showDialog) {
          showDialog();
        }
    };

    const defaultCards = [
        { imageSrc: Preprocess, title: "Annotations Tools", showDialog: true },
        { imageSrc: BloodCells, title: "Cellular Detection ", link: "/TrainCell"},
        { imageSrc: Tissue, title: "Tissue Classification", link: "/TrainTissue" },
    ];

    return (
        <Container>
            <TwoColumnContainer>
            
        <Heading>{heading}</Heading>
        {description && <Description>{description}</Description>}
               
                <VerticalSpacer />
                {defaultCards.map((card, i) => (
                    <Column key={i}>
                       <Card>
              <span className="imageContainer">
                <img src={card.imageSrc || defaultCardImage} alt="" />
              </span>
              <span className="textContainer">
                {/* Check if the card has a showDialog property */}
                {card.showDialog ? (
                  <span className="title" onClick={showDialog}>{card.title}</span>
                ) : (
                  // If card has a link, use the Link component to wrap the title
                  <Link to={card.link} style={{ textDecoration: 'none' }}>
                    <span className="title">{card.title}</span>
                  </Link>
                )}
              </span>
            </Card>
          </Column>
        ))}
      </TwoColumnContainer>
      {isDialogOpen && <AnnotationTypeDialoge onClose={hideDialog} />}
      <DecoratorBlob />
    </Container>
    );
};


