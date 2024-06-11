import React, { useState, useRef, useEffect } from 'react';
import TrainCell from './TrainCell';
import TrainTissue from './TrainTissue';
import styled from 'styled-components';
import Header from "components/headers/light";
import Footer from "components/footers/FiveColumnWithBackground";
import cellImage from './resnet.png';
import tissueImage from './unet.png';

const PageContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
`;

const SelectTrainingTypeContainer = styled.div`
  display: flex;
  justify-content: center;
  margin-bottom: 30px;
`;

const SelectWrapper = styled.div`
  position: relative;
  width: 300px;
`;

const SelectTrainingType = styled.select`
  width: 100%;
  padding: 10px;
  font-size: 16px;
  border: 1px solid #ccc;
  border-radius: 30px;
  background: white;
  color: #333;
  appearance: none;
  cursor: pointer;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);

  &:focus {
    border-color: #6415FF;
    box-shadow: 0 0 10px rgba(100, 21, 255, 0.5);
    outline: none;
  }

  &:hover {
    background: #f0f0f0;
  }
`;

const SelectArrow = styled.div`
  position: absolute;
  top: 50%;
  right: 15px;
  pointer-events: none;
  transform: translateY(-50%);
  font-size: 16px;
  color: #333;
`;

const Title = styled.h1`
  text-align: center;
  margin-bottom: 40px;
  color: #333;
  font-size: 2.5rem;
  background: -webkit-linear-gradient(#6415FF, #91BDFE);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
`;

const Card = styled.div`
  background: #fff;
  border: 1px solid #e5e5e5;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  cursor: pointer;
  max-width: 1000px;
  margin: 0 auto 20px;

  &:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.2);
  }
`;

const InitialContent = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  margin-top: 40px;
`;

const ContentRow = styled.div`
  display: flex;
  justify-content: space-around;
  align-items: center;
  margin-bottom: 40px;
  width: 100%;

  &:nth-child(odd) {
    flex-direction: row-reverse;
  }
`;

const ImageContainer = styled.div`
  width: 30%;
  img {
    width: 100%;
    border-radius: 12px;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
  }
`;

const Description = styled.div`
  width: 60%;
  h2 {
    font-size: 1.8rem;
    color: #333;
    margin-bottom: 20px;
  }
  p {
    font-size: 1rem;
    color: #666;
    line-height: 1.5;
  }
`;

const TrainModel = () => {
  const [trainingType, setTrainingType] = useState('');
  const cellRef = useRef(null);
  const tissueRef = useRef(null);

  const handleTrainingTypeChange = (e) => {
    setTrainingType(e.target.value);
  };

  useEffect(() => {
    if (trainingType === 'cell' && cellRef.current) {
      cellRef.current.scrollIntoView({ behavior: 'smooth' });
    } else if (trainingType === 'tissue' && tissueRef.current) {
      tissueRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [trainingType]);

  return (
    <>
      <Header />
      <PageContainer>
        <Title>Train Your Model</Title>
        <SelectTrainingTypeContainer>
          <SelectWrapper>
            <SelectTrainingType value={trainingType} onChange={handleTrainingTypeChange}>
              <option value="">Select Training Type</option>
              <option value="cell">Cell Level</option>
              <option value="tissue">Tissue Level</option>
            </SelectTrainingType>
            <SelectArrow>▼</SelectArrow>
          </SelectWrapper>
        </SelectTrainingTypeContainer>
        <InitialContent>
          <ContentRow>
            <ImageContainer>
              <img src={cellImage} alt="Cell Training" />
            </ImageContainer>
            <Description>
              <h2>Cell Level Training</h2>
              <p>Use our advanced tools to train models on a cellular level. Upload your datasets, configure training parameters, and get accurate results tailored to your research needs.</p>
            </Description>
          </ContentRow>
          <ContentRow>
            <ImageContainer>
              <img src={tissueImage} alt="Tissue Training" />
            </ImageContainer>
            <Description>
              <h2>Tissue Level Training</h2>
              <p>Leverage our platform to train models on tissue samples. Our intuitive interface and powerful algorithms ensure high-quality predictions and insights.</p>
            </Description>
          </ContentRow>
        </InitialContent>
        {trainingType === 'cell' && (
          <div ref={cellRef}>
            <TrainCell />
          </div>
        )}
        {trainingType === 'tissue' && (
          <div ref={tissueRef}>
            <TrainTissue />
          </div>
        )}
      </PageContainer>
      <Footer />
    </>
  );
};

export default TrainModel;
