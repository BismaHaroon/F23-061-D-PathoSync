import React, {useRef} from "react";
import tw from "twin.macro";
import { css } from "styled-components/macro"; //eslint-disable-line
import AnimationRevealPage from "helpers/AnimationRevealPage.js";
import Hero from "components/hero/TwoColumnWithInput.js";
import Features from "components/features/ThreeColWithSideImage.js";
import MainFunctions from "components/MainFunctions/MainFunctions";

import FAQ from "components/faqs/SingleCol.js";

import Footer from "components/footers/FiveColumnWithBackground.js";



export default () => {

  const featuresRef = useRef(null);
  const scrollToFeatures = () => {
    window.scrollBy({
      top: 600, // Scroll down 100 pixels
      behavior: 'smooth' // Optional: Define smooth scrolling
    });
  };
  
  
  const Subheading = tw.span`uppercase tracking-widest font-bold text-primary-500`;
  const HighlightedText = tw.span`text-primary-500`;
  
  return (
    <AnimationRevealPage>
      <Hero roundedHeaderButton={true} />
      

      <MainFunctions 
      //  subheading={<Subheading>Features</Subheading>}
       heading={
         <>
          
           <HighlightedText  > Features </HighlightedText> Offered 
         </>
       }
      />
      <button onClick={scrollToFeatures} style={{
  background: 'linear-gradient(to right, #6B46C1, #9B59B6)',
  color: 'black',
  padding: '8px 8px',
  borderRadius: '20px',
  left: '50%',
  transform: 'translateX(-50%)',
  zIndex: 1000,
  marginTop: '-80px',
  marginLeft: '720px',
  height: '40px',
  width: '40px'

}} >

  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" id="ChevronDown"><path fill="#ffffff" d="M3.95041,6.48966 C4.23226,6.18613 4.70681,6.16856 5.01034,6.45041 L8,9.22652 L10.9897,6.45041 C11.2932,6.16856 11.7677,6.18613 12.0496,6.48966 C12.3315,6.7932 12.3139,7.26775 12.0103,7.5496 L8.51034,10.7996 C8.22258,11.0668 7.77743,11.0668 7.48966,10.7996 L3.98966,7.5496 C3.68613,7.26775 3.66856,6.7932 3.95041,6.48966 Z" class="color212121 svgShape"></path></svg>
</button>

      <Features
        subheading={<Subheading>Features</Subheading>}
        heading={
          <>
           
           Unlock  <HighlightedText> Powerful </HighlightedText>Capabilities
          </>
        }
      />
      

    
      
      <FAQ
  subheading={<Subheading>Frequently Asked Questions</Subheading>}
  heading={
    <>
      Have <HighlightedText>Questions about PathoSync?</HighlightedText>
    </>
  }
  faqs={[
    {
      question: "What is PathoSync?",
      answer:
        "PathoSync is a revolutionary digital pathology practitioners' tool. It offers a user-friendly web-based framework for medical image analysis, annotation, and AI model training, enhancing the accuracy of diagnostics and treatment strategies."
    },
    {
      question: "How does PathoSync simplify image analysis?",
      answer:
        "PathoSync simplifies image uploading, preprocessing, cellular annotations, and tissue region classification through its intuitive interface. It eliminates the need for deep learning expertise among pathologists, making it accessible and efficient."
    },
    {
      question: "Can I customize the annotations in PathoSync?",
      answer:
        "Yes, PathoSync allows precise marking and labeling of individual cells as well as outlining and labeling larger tissue regions. The annotations are fully customizable and editable to meet the specific requirements of medical practitioners."
    },
    {
      question: "What technologies are used in building PathoSync?",
      answer:
        "PathoSync is built using cutting-edge technologies, including React for the frontend, Python and Django for backend development, PyTorch/TensorFlow for deep learning, and AWS for hosting and scalability."
    },

  ]}
/>

   
      <Footer />
    </AnimationRevealPage>
  );
}
