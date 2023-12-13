import React from "react";
import GlobalStyles from 'styles/GlobalStyles';
import { css } from "styled-components/macro"; //eslint-disable-line




import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import LandingPage from "demos/LandingPage";
import ImageUploadPage from "demos/ImageUpload";
import AnnotateCell from "demos/AnnotateCell";
import AnnotateTissue from "demos/AnnotateTissue";
import AnnotateSAM from "demos/AnnotateSAM";
export default function App() {



  return (
    <>
      <GlobalStyles />
      <Router>
        <Routes>
         
          <Route path="/" element={<LandingPage />} />
        
          <Route path="/:option" element={<ImageUploadPage />} />
          <Route path="/AnnotateCell" element={<AnnotateCell/>} />
          <Route path="/AnnotateTissue/:processedImage" element={<AnnotateTissue/>} />
          <Route path="/AnnotateSAM" element={<AnnotateSAM/>} />
        </Routes>
      </Router>
    </>
  );
}


