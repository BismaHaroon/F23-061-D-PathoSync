import React from 'react';
import tw from 'twin.macro';
import Header from "components/headers/light";
import Footer from "components/footers/FiveColumnWithBackground";
import { useLocation } from 'react-router-dom';
import { SectionHeading, Subheading as SubheadingBase } from "components/misc/Headings.js";
import { SectionDescription } from "components/misc/Typography.js";

const GridContainer = tw.div`flex flex-wrap justify-center mx-auto`; 
const ImagePreview = tw.img`max-w-xs h-auto rounded shadow-lg m-1`;
const Container = tw.div`relative text-center`;
const Subheading = tw(SubheadingBase)`my-8 text-center text-4xl`;
const Heading = tw.h1`text-5xl font-bold mb-8 text-transparent bg-gradient-to-r from-purple-500 to-blue-500 bg-clip-text`;
const Description = tw(SectionDescription)`w-full text-center`;
const OverlayContainer = tw.div`w-full mt-8 flex justify-center items-center`; 
const OverlayImage = tw.img`max-w-full h-auto rounded shadow-lg`;
const VerticalSpacer = tw.div`mt-2 w-full`;

const TissuePredictionResults = (props) => {
  const location = useLocation();
  const { patchPlotFiles, overlayPlot } = location.state || { patchPlotFiles: [], overlayPlot: null };
  const baseURL = "http://127.0.0.1:5000/";

  // Check if patchPlotFiles is not null and has at least one plot file
  const hasPatchPlotFiles = patchPlotFiles && Array.isArray(patchPlotFiles) && patchPlotFiles.length > 0;

  return (
    <Container>
      <Header />
      <Heading>Tissue Prediction</Heading>
      
      {/* <Description>This is a placeholder description for the tissue prediction results.</Description> */}
      <VerticalSpacer />

      {hasPatchPlotFiles ? (
        
        <GridContainer>
          <Heading>Patch Plotting</Heading>
          {patchPlotFiles.map((plotFile, index) => (
            <div key={index} css={tw`p-4`}>
              <ImagePreview src={`${baseURL}download_patch_plot/${plotFile}`} alt={`Patch Plot ${index}`} />
            </div>
          ))}
        </GridContainer>
      ) : (
        <p></p> // You can customize this message or remove it entirely
      )}

      <Subheading>Overlay Plotting</Subheading>
      <OverlayContainer>
        <a href={`${baseURL}download_overlay_plot/${overlayPlot}`} download="Overlay_Plot.png">
          <OverlayImage src={`${baseURL}download_overlay_plot/${overlayPlot}`} alt="Overlay Plot" />
        </a>
      </OverlayContainer>

      <Footer />
    </Container>
  );
};

export default TissuePredictionResults;
