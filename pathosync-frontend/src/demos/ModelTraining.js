import React from "react";
import Header from "components/headers/light";
import Footer from "components/footers/FiveColumnWithBackground";
import "./ModelTraining.css";

const ModelTraining = () => {
  return (
    <>
      <Header />
      <div className="container">
        <h1 className="heading">Model Training</h1>
        <div className="button-container">
          <button className="btn cube cube-hover" type="button">
            <div className="bg-top">
              <div className="bg-inner"></div>
            </div>
            <div className="bg-right">
              <div className="bg-inner"></div>
            </div>
            <div className="bg">
              <div className="bg-inner"></div>
            </div>
            <div className="text">Cell Detection</div>
          </button>
          <button className="btn cube cube-hover" type="button">
            <div className="bg-top">
              <div className="bg-inner"></div>
            </div>
            <div className="bg-right">
              <div className="bg-inner"></div>
            </div>
            <div className="bg">
              <div className="bg-inner"></div>
            </div>
            <div className="text">Tissue Classification</div>
          </button>
        </div>
      </div>
      <div className="footer-container">
        <Footer />
      </div>
    </>
  );
};

export default ModelTraining;
