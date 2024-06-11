import React from "react";
import GlobalStyles from 'styles/GlobalStyles';
import { css } from "styled-components/macro"; //eslint-disable-line
import TrainModel from "demos/TrainModel";
import PredictionHub from "demos/PredictModel"




import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import LandingPage from "demos/LandingPage";
import ImageUploadPage from "demos/ImageUpload";
import AnnotateCell from "demos/AnnotateCell";
import AnnotateTissue from "demos/AnnotateTissue";
import AnnotateSAM from "demos/AnnotateSAM";
import PredictTissue from "demos/PredictTissue";
import PredictCell from "demos/PredictCell";
import ModelTraining from "demos/ModelTraining";
import TrainTissue from "demos/TrainTissue";
import TrainCell from "demos/TrainCell";
import CellPredictResult from "demos/CellPredictionResult";
import TissuePredictionResults from "demos/TissuePredictionResults";
import WSIUpload from "demos/WSIUpload";
import AnnotateWSI from "demos/AnnotateWSI";
import AnnotateWSIPatch from "demos/AnnotateWSIPatch";
import CreateProject from "demos/CreateProject";
export default function App() {



  return (
    <>
      <GlobalStyles />
      <Router>
        <Routes>
         
          <Route path="/" element={<LandingPage />} />
          <Route path="/WSIUpload" element={<WSIUpload/>} />
          <Route path="/AnnotateWSI" element={<AnnotateWSI/>} />
          <Route path="/AnnotateWSIPatch" element={<AnnotateWSIPatch/>} />
          <Route path="/:option" element={<ImageUploadPage />} />
          <Route path="/AnnotateCell" element={<AnnotateCell/>} />
          <Route path="/AnnotateTissue" element={<AnnotateTissue/>} />
          <Route path="/AnnotateSAM" element={<AnnotateSAM/>} />
          <Route path="/PredictTissue" element={<PredictTissue/>} />
          <Route path="/PredictCell" element={<PredictCell/>} />
          <Route path="/ModelTraining" element={<ModelTraining/>} />
          <Route path="/TrainTissue" element={<TrainTissue/>} />
          <Route path="/TrainCell" element={<TrainCell/>} />
          <Route path="/cell-predict-result" element={<CellPredictResult/>} />
          <Route path="/tissue-patch-results" element={<TissuePredictionResults/>} />
          <Route path="CreateProject" element={<CreateProject/>} />
          <Route path="/Predictions" element={<PredictionHub/>} />
          <Route path="/TrainModel" element={<TrainModel/>} />
        </Routes>
      </Router>
    </>
  );
}
