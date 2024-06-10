import React from "react";
import tw from "twin.macro";
import { css } from "styled-components/macro"; //eslint-disable-line
import AnimationRevealPage from "helpers/AnimationRevealPage.js";
import Hero from "components/hero/TwoColumnWithInput.js";
import Features from "components/features/ThreeColWithSideImage.js";
<<<<<<< HEAD
import MainFunctions from "components/MainFunctions/MainFunctions";
=======

>>>>>>> 8655ceccc37e8fd8d0bdcbd17d190dc036418d41

import FAQ from "components/faqs/SingleCol.js";

import Footer from "components/footers/FiveColumnWithBackground.js";



export default () => {
  const Subheading = tw.span`uppercase tracking-widest font-bold text-primary-500`;
  const HighlightedText = tw.span`text-primary-500`;

  return (
    <AnimationRevealPage>
      <Hero roundedHeaderButton={true} />
<<<<<<< HEAD
      <MainFunctions 
       
       heading={
         <>
          
           <HighlightedText> Features </HighlightedText> Offered 
         </>
       }
      />
=======
>>>>>>> 8655ceccc37e8fd8d0bdcbd17d190dc036418d41
      <Features
        subheading={<Subheading>Features</Subheading>}
        heading={
          <>
           
           Unlock  <HighlightedText> Powerful </HighlightedText>Capabilities
          </>
        }
      />
      
<<<<<<< HEAD
    
=======
>>>>>>> 8655ceccc37e8fd8d0bdcbd17d190dc036418d41
      
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
    {
      question: "Is my patient data secure with PathoSync?",
      answer:
        "Absolutely. PathoSync prioritizes data privacy and security. Patient data is securely stored in compliance with relevant healthcare data privacy regulations, ensuring confidentiality and adherence to industry standards."
    }
  ]}
/>

   
      <Footer />
    </AnimationRevealPage>
  );
}
