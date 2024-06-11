import React from "react";
import styled from "styled-components";
import tw from "twin.macro";
import { SectionHeading, Subheading as SubheadingBase } from "components/misc/Headings.js";
import { SectionDescription } from "components/misc/Typography.js";

import defaultCardImage from "images/shield-icon.svg";

import { ReactComponent as SvgDecoratorBlob3 } from "images/svg-decorator-blob-3.svg";

import Annotate from "images/annotate.svg"
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
  ${tw`flex flex-col sm:flex-row items-center sm:items-start text-center sm:text-left h-full mx-4 px-2 py-8`}
  .imageContainer {
    ${tw`border text-center rounded-full p-5 flex-shrink-0`}
    img {
      ${tw`w-10 h-10`}
    }
  }

  .textContainer {
    ${tw`sm:ml-4 mt-4 sm:mt-2`}
  }

  .title {
    ${tw`mt-4 tracking-wide font-bold text-2xl leading-none cursor-pointer`}
  }

  .description {
    ${tw`mt-1 sm:mt-4 font-medium text-secondary-100 leading-loose`}
  }
`;

const DecoratorBlob = styled(SvgDecoratorBlob3)`
  ${tw`pointer-events-none absolute right-0 bottom-0 w-64 opacity-25 transform translate-x-32 translate-y-48 `}
`;

export default ({
  cards = null,
  heading = "Amazing Features",
  subheading = "Features",
  description = "Unlock Powerful Capabilities: PathoSync offers simplified image uploading, automated image preprocessing, precise cellular and tissue annotations, and seamless model training."
}) => {
  const defaultCards = [
    { imageSrc: Annotate, title: "Annotations Tools", description: "Enhance image quality and remove noise for optimized analysis." },
    { imageSrc: BloodCells, title: "Cellular Detection", description: "Precise marking and labeling of individual cells using advanced algorithms." },
    { imageSrc: Tissue, title: "Tissue Classification", description: "Annotations and identification of different tissue types for enhanced analysis." },
    { imageSrc: ModelTraining, title: "Cellular Prediction", description: "Upload annotated images for AI model training, utilizing advanced frameworks." },
    { imageSrc: ModelTraining, title: "Tissue Prediction", description: "Predictions and insights on tissue images using trained AI models." }
  ];

  if (!cards) cards = defaultCards;

  return (
    <Container>
      <TwoColumnContainer>
        {subheading && <Subheading>{subheading}</Subheading>}
        <Heading>{heading}</Heading>
        {description && <Description>{description}</Description>}
        <VerticalSpacer />
        {cards.map((card, i) => (
          <Column key={i}>
            <Card>
              <span className="imageContainer">
                <img src={card.imageSrc || defaultCardImage} alt="" />
              </span>
              <span className="textContainer">
                <span className="title">{card.title || "Fully Secure"}</span>
                {/* <p className="description">{card.description || "Lorem ipsum donor amet siti ceali ut enim ad minim veniam, quis nostrud."}</p> */}
              </span>
            </Card>
          </Column>
        ))}
      </TwoColumnContainer>
      <DecoratorBlob />
    </Container>
  );
};
