import React from "react";
import GlobalStyles from 'styles/GlobalStyles';
import { css } from "styled-components/macro"; //eslint-disable-line




import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import LandingPage from "demos/LandingPage";
import ImageUploadPage from "demos/ImageUpload";
import AnnotateCell from "demos/AnnotateCell";
import AnnotateTissue from "demos/AnnotateTissue";
import AnnotateSAM from "demos/AnnotateSAM";
import WSIUpload from "demos/WSIUpload";
import AnnotateWSI from "demos/AnnotateWSI";
import AnnotateWSIPatch from "demos/AnnotateWSIPatch";
import ModelTrain from "demos/ModelTrain";
import PredictTissue from "demos/PredictTissue";
import PredictCell from "demos/PredictCell";
import TissuePredictionResults from "demos/TissuePredictionResults";
import CellPredictionResult from "demos/CellPredictionResult";
import TrainCell from "demos/TrainCell";
import TrainTissue from "demos/TrainTissue";
import TrainModel from "demos/TrainModel";
import PredictionHub from "demos/PredictModel"
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
          <Route path="/WSIUpload" element={<WSIUpload/>} />
          <Route path="/AnnotateWholeWSI" element={<AnnotateWSI/>} />
          <Route path="/AnnotateWSIPatch" element={<AnnotateWSIPatch/>} />
          <Route path="/ModelTrain" element={<ModelTrain/>} />
          <Route path="/PredictTissue" element={<PredictTissue/>} />
          <Route path="/PredictCell" element={<PredictCell/>} />
          <Route path="/tissue-patch-results" element={<TissuePredictionResults/>} />
          <Route path="/cell-predict-result" element={<CellPredictionResult/>} />
          <Route path="/TrainCell" element={<TrainCell/>} />
          <Route path="/TrainTissue" element={<TrainTissue/>} />
          <Route path="/Predictions" element={<PredictionHub/>} />
          <Route path="/TrainModel" element={<TrainModel/>} />
          {/* <Route path="/ModelTraining" element={<ModelTraining/>} /> */}
        </Routes>
      </Router>
    </>
  );
}


