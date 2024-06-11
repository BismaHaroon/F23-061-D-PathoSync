import React, { useRef } from "react";
import styled, { createGlobalStyle } from "styled-components";
import tw from "twin.macro";
//eslint-disable-next-line
import { css } from "styled-components/macro";

import Header from "../headers/light.js";

import { ReactComponent as SvgDecoratorBlob1 } from "../../images/svg-decorator-blob-1.svg";
import DesignIllustration from "../../images/design-illustration-2.svg";
import Doctorhero from "../../images/doctor-hero.jpg"
import BGimage from "../../images/bgImage.png"
import CustomersLogoStripImage from "../../images/customers-logo-strip.png";

const Container = styled.div`
  ${tw`relative min-h-screen bg-cover bg-center`}
  background-image: url(${BGimage});
  background-size: cover; // Ensures the background covers the entire element
  
`;

const Content = styled.div`
  ${tw`max-w-screen-xl mx-auto py-20 md:py-24 px-4 lg:px-8`}
`;

const GlobalStyle = createGlobalStyle`
  body {
    background-color: #ffffff; /* Adjust the color to the shade of purple you prefer */
  }
`;

const TwoColumn = tw.div`flex flex-col lg:flex-row lg:items-center max-w-screen-xl mx-auto py-20 md:py-24`;
const LeftColumn = tw.div`relative lg:w-5/12 text-center max-w-lg mx-auto lg:max-w-none lg:text-left`;
const RightColumn = tw.div`relative mt-12 lg:mt-0 flex-1 flex flex-col justify-center lg:self-end`;

const Heading = tw.h1`font-bold text-3xl md:text-3xl lg:text-4xl xl:text-5xl text-gray-900 leading-tight`;
const Paragraph = tw.p`my-5 lg:my-8 text-base xl:text-lg`;

const Actions = styled.div`
  ${tw`relative max-w-md text-center mx-auto lg:mx-0`}
  input {
    ${tw`sm:pr-48 pl-8 py-4 sm:py-5 rounded-full border-2 w-full font-medium focus:outline-none transition duration-300  focus:border-primary-500 hover:border-gray-500`}
  }
  button {
    ${tw`w-full sm:absolute right-0 top-0 bottom-0 bg-primary-500 text-gray-100 font-bold mr-2 my-4 sm:my-2 rounded-full py-4 flex items-center justify-center sm:w-40 sm:leading-none focus:outline-none hover:bg-primary-900 transition duration-300`}
  }
`;

const IllustrationContainer = tw.div`flex justify-center lg:justify-end items-center`;

// Random Decorator Blobs (shapes that you see in background)
const DecoratorBlob1 = styled(SvgDecoratorBlob1)`
  ${tw`pointer-events-none opacity-5 absolute left-0 bottom-0 h-64 w-64 transform -translate-x-2/3 -z-10`}
`;



export default ({ roundedHeaderButton }) => {
  const scrollToFeatures = () => {
    window.scrollBy({
      top: 790, // Scroll down 100 pixels
      behavior: 'smooth' // Optional: Define smooth scrolling
    });
  };
  return (
    <>
    <GlobalStyle/>
      <Header roundedHeaderButton={roundedHeaderButton} />
      <Container style={{marginTop: '-100px'}}>
        <TwoColumn>
          <LeftColumn>
            <Heading>
             <span style={{fontSize:'100px', textAlign:'right'}} tw="text-primary-500">PathoSync </span>
              Enhancing Pathology with Synchronized AI
            </Heading>
            
            
          
          </LeftColumn>
          <RightColumn>
          <Paragraph style={{color:'#ffffff', textAlign: 'justify'}}>
          <br/><br/>Advance Your Pathology Diagnostics With PathoSync. Seamlessly Upload,<br/> Enhance, And Analyze Medical Images With Precision. Identify Cellular Structures<br/> And Classify Tissues Effortlessly For Detailed And Accurate Assessments. <br/>
            Enrich Your Analytic Capabilities With Our State-Of-The-Art Model Training Tools,<br/> Designed To Boost Diagnostic Accuracy And Efficiency.
            </Paragraph>
          </RightColumn>
          
        </TwoColumn>
        <button onClick={scrollToFeatures} style={{
  background: 'linear-gradient(to right, #6B46C1, #9B59B6)',
  color: 'black',
  padding: '8px 8px',
  borderRadius: '20px',
  left: '50%',
  transform: 'translateX(-50%)',
  zIndex: 1000,
  marginTop: '200px',
  marginLeft: '720px',
  height: '40px',
  width: '40px'

}} >

  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" id="ChevronDown"><path fill="#ffffff" d="M3.95041,6.48966 C4.23226,6.18613 4.70681,6.16856 5.01034,6.45041 L8,9.22652 L10.9897,6.45041 C11.2932,6.16856 11.7677,6.18613 12.0496,6.48966 C12.3315,6.7932 12.3139,7.26775 12.0103,7.5496 L8.51034,10.7996 C8.22258,11.0668 7.77743,11.0668 7.48966,10.7996 L3.98966,7.5496 C3.68613,7.26775 3.66856,6.7932 3.95041,6.48966 Z" class="color212121 svgShape"></path></svg>
</button>
        <DecoratorBlob1 />
      </Container>
    </>
  );
};

