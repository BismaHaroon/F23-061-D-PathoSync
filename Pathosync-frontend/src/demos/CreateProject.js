import './CreateProjectStyles.css';
import Header from "components/headers/light";
import { useNavigate } from 'react-router-dom';
import React, { useState, useEffect } from 'react';
import Carousel from 'react-multi-carousel';
import 'react-multi-carousel/lib/styles.css';
import CustomCarousel from './CustomCarousel';
function CreateProject() {
  const [projectName, setProjectName] = useState('');
  const [datasetType, setDatasetType] = useState('cell');
  const [classes, setClasses] = useState([{ name: 'Cell Images' }, { name: 'Masks' }]);
  const [showPopup, setShowPopup] = useState(false);
  const [selectedProject, setSelectedProject] = useState(null);
  const [selectedImages, setSelectedImages] = useState([]);

  const [projectType, setProjectType] = useState('');
  const navigate = useNavigate();

  const handleProjectNameChange = (e) => {
    setProjectName(e.target.value);
  };

  const handleDatasetTypeChange = (e) => {
    setDatasetType(e.target.value);
    switch (e.target.value) {
      case 'cell':
        setClasses([{ name: 'Cell Images' }, { name: 'Masks' }]);
        break;
      case 'tissue':
      case 'WSI':
        setClasses([{ name: 'Annotated' }, { name: 'Unannotated' }]);
        break;
      default:
        setClasses([]);
    }
  };

  const handleAddClass = () => {
    setClasses(classes.concat([{ name: '' }]));
  };

  const handleRemoveClass = (index) => {
    setClasses(classes.filter((_, i) => i !== index));
  };

  const handleClassNameChange = (index, event) => {
    const newClasses = classes.map((cls, clsIndex) => {
      if (index === clsIndex) {
        return { ...cls, name: event.target.value };
      }
      return cls;
    });
    setClasses(newClasses);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const projectData = {
        name: projectName,
        numClasses: classes.length,
        datasetType: datasetType,
        classNames: classes.map(cls => cls.name)
    };

    try {
        const response = await fetch('http://localhost:5000/projects/create', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(projectData)
        });
        if (!response.ok) {
            throw new Error(`HTTP status ${response.status}`);
        }
        const result = await response.json();
        console.log('Project created:', result.project);
        alert(`Project created successfully! ID: ${result.project._id}`);

        // Update the projects list with the new project
        setProjects(prevProjects => [...prevProjects, {
            id: result.project._id,
            name: result.project.name,
            datasetType: result.project.datasetType,
            classNames: result.project.classNames,
            numClasses: result.project.numClasses,
            images: result.project.images || []
        }]);

    } catch (error) {
        console.error('Error creating project:', error);
        alert('Failed to create project.');
    }
  };

//Project List
const [projects, setProjects] = useState([]);

  useEffect(() => {
    fetchProjects();
  }, []);

  const fetchProjects = async () => {
    try {
      const response = await fetch('http://localhost:5000/projects');
      if (!response.ok) {
        throw new Error('Failed to fetch projects');
      }
      const projects = await response.json();
      setProjects(projects);
    } catch (error) {
      console.error('Error fetching projects:', error);
    }
  };


  const handleFileUpload = async (projectId, files) => {
    const formData = new FormData();
    for (const file of files) {
        formData.append('file', file);
    }

    try {
        const response = await fetch(`http://localhost:5000/projects/${projectId}/upload`, {
            method: 'POST',
            body: formData,
        });
        if (!response.ok) {
            throw new Error('Failed to upload files');
        }
        const result = await response.json();
        alert('Files uploaded successfully');

        // Update the projects list with the new images
        setProjects(prevProjects => prevProjects.map(project => {
            if (project.id === projectId) {
                return { ...project, images: [...project.images, ...result.images] };
            }
            return project;
        }));

    } catch (error) {
        console.error('Error uploading files:', error);
        alert('Failed to upload files');
    }
  };
// Annotations Choice
const handleAnnotateClick = (project) => {
  if (selectedImages.length === 0) {
    alert('Please select images first.');
    return;
  }

  setSelectedImages(selectedImages);
  if (project.datasetType === 'tissue') {
    navigate('/AnnotateTissue', { state: { project_name: project.name, images: selectedImages, classNames: project.classNames } });
  } else if (project.datasetType === 'cell') {
    setSelectedProject(project);
    setShowPopup(true);
  }
};

const handleAnnotationChoice = async (choice) => {
  setShowPopup(false);
  if (selectedImages.length === 0) {
    alert('Please select images first.');
    return;
  }

  const requestData = {
    project_name: selectedProject.name,
    images: selectedImages
  };

  try {
    const endpoint = '/get_latest_nuclick'; // Use the new combined endpoint
    const response = await fetch(`http://localhost:5000${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestData)
    });

    if (!response.ok) {
      throw new Error(`Failed to start annotation: ${response.statusText}`);
    }

    const responseData = await response.json();
    console.log('Response Data:', responseData);

    if (choice === 'Point') {
      navigate('/AnnotateCell', { state: { images: responseData.images, project_name: responseData.project_name } });
    } else if (choice === 'SAM') {
      navigate('/AnnotateSAM', { state: { images: responseData.images,  project_name: responseData.project_name } });
    }

  } catch (error) {
    console.error('Error starting annotation:', error);
    alert(`Failed to start annotation: ${error.message}`);
  }
};



const handleImageSelect = (image) => {
  setSelectedImages(prevSelectedImages => {
    if (prevSelectedImages.includes(image)) {
      return prevSelectedImages.filter(img => img !== image);
    } else {
      return [...prevSelectedImages, image];
    }
  });
};


  return (
    <div>
      <Header />
      <div className="createProjectContainer">
        <h1>Create Project</h1>
        <form onSubmit={handleSubmit}>
          <div className="label-input-group">
            <label>
              Project Name:
              <input type="text" value={projectName} onChange={handleProjectNameChange} required placeholder="Enter project name" />
            </label>
          </div>
          <div className="label-input-group">
            <label>
              Dataset Type:  
              <select value={datasetType} onChange={handleDatasetTypeChange}>
                <option value="cell">Cell</option>
                <option value="tissue">Tissue</option>
                <option value="WSI">Whole Slide Image (WSI)</option>
              </select>
            </label>
          </div>
          <div>
            {classes.map((cls, index) => (
              <div key={index} className="label-input-group">
                <label>
                  Class Name {index + 1}:
                  <input type="text" value={cls.name} onChange={(e) => handleClassNameChange(index, e)} required placeholder="Enter class name" />
                  <button type="button" onClick={() => handleRemoveClass(index)} className="remove-btn">−</button>
                </label>
              </div>
            ))}
            <button type="button" onClick={handleAddClass} className="add-btn">Add Class</button>
          </div>
          <button type="submit" className="create-project-btn">Create Project</button>
        </form>
      </div>
      <div className="createProjectContainer">
        <h1>Projects Repository</h1>
        {projects.map(project => (
          <div key={project.id} className="projectContainer">
            <p><strong>Project ID:</strong> <span>{project.id}</span></p>
            <p><strong>Project Name:</strong> <span>{project.name}</span></p>
            <p><strong>Dataset Type:</strong> <span>{project.datasetType}</span></p>
            <p><strong>Classes:</strong> <span>{project.classNames.join(', ')}</span></p>
            <input type="file" multiple onChange={(e) => handleFileUpload(project.id, e.target.files)} />
            <CustomCarousel images={project.images} onSelect={handleImageSelect} selectedImages={selectedImages} />
            <button onClick={() => handleAnnotateClick(project)} className="add-btn">Annotate</button>
          </div>
        ))}
      </div>
      {showPopup && (
        <div className="popup">
          <div className="popup-content">
            <h3>Select Annotation Type</h3>
            <button onClick={() => handleAnnotationChoice('SAM')}>SAM Annotation</button>
            <button onClick={() => handleAnnotationChoice('Point')}>Point Annotation</button>
            <button onClick={() => setShowPopup(false)}>Cancel</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default CreateProject;
