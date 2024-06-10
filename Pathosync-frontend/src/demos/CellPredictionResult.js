import React from 'react';
import { useLocation } from 'react-router-dom';
import styled from 'styled-components';
import tw from 'twin.macro';
import Header from "components/headers/light";
import Footer from "components/footers/FiveColumnWithBackground";

const PageContainer = styled.div`
  ${tw`flex flex-col items-center justify-center min-h-screen py-8`}
`;

const ResultsContainer = styled.div`
  ${tw`max-w-4xl w-full p-8 shadow-lg rounded-lg bg-white`}
  margin: 2rem;
  text-align: center;
`;

const Heading = tw.h1`text-5xl font-bold mb-8 text-transparent bg-gradient-to-r from-purple-500 to-blue-500 bg-clip-text`;

const ImageContainer = styled.div`
  ${tw`overflow-hidden`}
  max-width: 90%;
  margin: auto;
`;

const Image = styled.img`
  ${tw`max-w-full h-auto rounded-lg`}
  max-height: 400px; /* Adjust the max-height to make the image smaller */
`;

const Text = styled.p`
  ${tw`text-xl mt-4`}
`;

const CellPredictResult = () => {
    const location = useLocation();
    const { resultData } = location.state;

    return (
        <>
          <Header />
          <PageContainer>
            <ResultsContainer>
              <Heading>Cell Prediction Results</Heading>
              <ImageContainer>
                <Image src={`http://127.0.0.1:5000/${resultData.segmented_image_filename}`} alt="Cell Prediction Result" />
              </ImageContainer>
            </ResultsContainer>
          </PageContainer>
          <Footer />
        </>
    );
};

export default CellPredictResult;
