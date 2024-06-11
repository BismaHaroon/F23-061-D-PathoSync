import './CreateProjectStyles.css';
import Header from "components/headers/light";
import { useNavigate } from 'react-router-dom';
import React, { useState, useEffect } from 'react';
import Carousel from 'react-multi-carousel';
import 'react-multi-carousel/lib/styles.css';
import CustomCarousel from './CustomCarousel';
import WSICarousel from './WSICarousel';
import axios from "axios";
function CreateProject() {
  const [projectName, setProjectName] = useState('');
  const [datasetType, setDatasetType] = useState('cell');
  const [classes, setClasses] = useState([{ name: 'Cell Images' }, { name: 'Masks' }]);
  const [showPopup, setShowPopup] = useState(false);
  const [selectedProject, setSelectedProject] = useState(null);
  const [selectedImages, setSelectedImages] = useState([]);
// WSI
const [wsiFile, setWSIFile] = useState(null);  // To store the WSI file
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
          images: result.project.images || [],
          patches: result.project.patches || []
        }]);
        
    } catch (error) {
        console.error('Error creating project:', error);
        alert('Failed to create project.');
    }
  };
  const handleWSIUpload = async (file, projectId, projectName) => { // Added projectName parameter
    if (file) {
      setWSIFile(file);

      try {
        // Request a SAS token from the backend
        const response = await axios.post('http://127.0.0.1:5000/get_sas_token', {
          filename: file.name,
          filetype: file.type,
        });

        const { url, token } = response.data;

        // Upload the file directly to Azure Blob Storage
        await axios.put(`${url}?${token}`, file, {
          headers: {
            'x-ms-blob-type': 'BlockBlob',
            'Content-Type': file.type,
          },
        });

        // Request patch creation
        const patchResponse = await axios.post('http://127.0.0.1:5000/create_patches', {
          filename: file.name,
          project_name: projectName  // Pass projectName to backend
        });

        if (patchResponse.status === 200) {
          alert('Patches created successfully');
          fetchProjects(); // Refresh projects to include the new patches
        } else {
          console.error('Error creating patches:', patchResponse);
          alert('Failed to create patches');
        }
      } catch (error) {
        console.error('Error uploading WSI file:', error);
        alert('Failed to upload WSI file.');
      }
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
      console.log('Projects fetched:', projects); // Log fetched projects
        projects.forEach(project => {
            console.log(`Project ${project.name} images:`, project.images);  // Log images
        });
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
  }else if (project.datasetType === 'WSI') {
      navigate('/AnnotateWSi', { state: { project_name: project.name, images: selectedImages, classNames: project.classNames } });
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

const WSIUploadDialog = ({ isOpen, onClose, onUpload }) => {
  if (!isOpen) return null;

  const handleFileChange = (event) => {
    onUpload(event.target.files[0]);
    onClose();
  };

  return (
    <div className="dialog-overlay">
      <div className="dialog-content">
        <h3>Upload Whole Slide Image (WSI)</h3>
        <input type="file" onChange={handleFileChange} />
        <button onClick={onClose}>Close</button>
      </div>
    </div>
  );
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

const handleDeleteProject = async (projectId) => {
  try {
    const response = await axios.delete(`http://localhost:5000/projects/${projectId}`);
    if (response.status === 200) {
      alert('Project deleted successfully');
      setProjects(prevProjects => prevProjects.filter(project => project.id !== projectId));
    } else {
      throw new Error('Failed to delete project');
    }
  } catch (error) {
    console.error('Error deleting project:', error);
    alert('Failed to delete project');
  }
};

useEffect(() => {
  fetchProjects();
}, []);


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
            
            {project.datasetType === 'WSI' ? (
              <div>
                <input type="file" onChange={(e) => handleWSIUpload(e.target.files[0], project.id, project.name)} />
                <div>
                  {project.images && project.images.length > 0 && (
                    <WSICarousel images={project.images} onSelect={handleImageSelect} selectedImages={selectedImages} />
                  )}
                </div>
                <button onClick={() => handleAnnotateClick(project)} className="add-btn">Annotate</button>
                <button onClick={() => handleDeleteProject(project.id)} className="add-btn" style={{ marginLeft: '1150px' }}>Delete Project</button>
              </div>
            ) : (
              <div>
                <input type="file" multiple onChange={(e) => handleFileUpload(project.id, e.target.files)} />
                <CustomCarousel images={project.images} onSelect={handleImageSelect} selectedImages={selectedImages} />
                <button onClick={() => handleAnnotateClick(project)} className="add-btn">Annotate</button>
                <button onClick={() => handleDeleteProject(project.id)} className="add-btn" style={{ marginLeft: '1150px' }}>Delete Project</button>
              </div>
            )}
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


const groupPatchesByImage = (patches) => {
  return patches.reduce((acc, patch) => {
    if (!acc[patch.image_filename]) {
      acc[patch.image_filename] = [];
    }
    acc[patch.image_filename].push(patch);
    return acc;
  }, {});
};
export default CreateProject;
