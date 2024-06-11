import React, { useState, useEffect, useRef } from 'react';
import { fabric } from 'fabric';
import Header from "components/headers/light";
import axios from 'axios';
import { useParams } from "react-router-dom";
import { useLocation } from 'react-router-dom';
import TissueCarousel from "./TissueCarousel";
import AnnotatedImagesDisplay from './AnnotatedImagesDisplay';
import { Link } from 'react-router-dom';

const hexToRGBA = (hex, alpha = 1) => {
  let r = parseInt(hex.slice(1, 3), 16),
      g = parseInt(hex.slice(3, 5), 16),
      b = parseInt(hex.slice(5, 7), 16);

  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

const App = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [canvas, setCanvas] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [labelText, setLabelText] = useState('');
  const [annotations, setAnnotations] = useState([]); // Store annotation objects
  const canvasRef = useRef(null);
  const [annotationColor, setAnnotationColor] = useState('#FF0000'); // Default color
  const { processedImage } = useParams();
  const [isLoading, setIsLoading] = useState(true);
  const [imageUrls, setImageUrls] = useState([]);
  const [selectedImages, setSelectedImages] = useState([]);
  const [imageAnnotations, setImageAnnotations] = useState({});
  const [selectedClass, setSelectedClass] = useState('');
  const [projectClasses, setProjectClasses] = useState([]);
  const [annotatedImages, setAnnotatedImages] = useState({});
  const [classImages, setClassImages] = useState({});

  console.log('processedImage:', processedImage);
  const [imageUrl, setImageUrl] = useState(null);
  useEffect(() => {
    setImageUrl(decodeURIComponent(processedImage));
  }, [processedImage]);


  const location = useLocation();
  const { project_name, images, classNames } = location.state || { project_name: '', images: [], classNames: [] };


  // Debugging: Log the received data
  console.log('Received data in frontend:', { images, project_name, classNames });


  useEffect(() => {
    const fetchImages = async () => {
      try {
        const urls = images.map(image => ({
          id: image.id,
          filename: image.filename,
          filepath: `http://localhost:5000/${image.filepath}`
        }));
        setImageUrls(urls);
      } catch (error) {
        console.error("Error fetching images:", error);
      } finally {
        setIsLoading(false);
      }
    };

    if (images.length > 0) {
      fetchImages();
    }
  }, [images]);

  const handleImageSelect = async (image) => {
    try {
      const sanitizedProjectName = project_name.replace(/\s+/g, '_');
      const response = await axios.post('http://127.0.0.1:5000/process-image-tissue', {
        image_path: image.filepath,
        project_name: sanitizedProjectName
      });
  
      const { image_path } = response.data;
  
      // Construct the URL for the normalized image
      const normalizedImageUrl = `http://localhost:5000/uploads/${sanitizedProjectName}_dataset/normalized_images/${image.filename}`;
      setSelectedImage(normalizedImageUrl);
    } catch (error) {
      console.error('Error processing image:', error);
    }
  };

  useEffect(() => {
    if (selectedImage) {
      setUploadedImage(selectedImage);
      // Clear canvas and reset annotations for the new image
      if (canvas) {
        canvas.clear();
        canvas.setBackgroundImage(null, canvas.renderAll.bind(canvas));
        setAnnotations([]);
      }
    }
  }, [selectedImage, canvas]);

  useEffect(() => {
    if (uploadedImage && canvasRef.current) {
      if (!canvas) {
        const newCanvas = new fabric.Canvas(canvasRef.current, {
          selection: true,
          backgroundColor: '#f0f0f0',
        });

        fabric.Image.fromURL(
          uploadedImage,
          img => {
            const aspectRatio = img.width / img.height;
            const canvasWidth = 800; // Adjust the width as needed
            const canvasHeight = canvasWidth / aspectRatio;

            newCanvas.setWidth(canvasWidth);
            newCanvas.setHeight(canvasHeight);

            newCanvas.setBackgroundImage(img, newCanvas.renderAll.bind(newCanvas), {
              scaleX: canvasWidth / img.width,
              scaleY: canvasHeight / img.height,
            });
          },
          { crossOrigin: 'anonymous' }
        );

        setCanvas(newCanvas);

        return () => {
          newCanvas.dispose();
          setCanvas(null);
        };
      } else {
        fabric.Image.fromURL(
          uploadedImage,
          img => {
            const aspectRatio = img.width / img.height;
            const canvasWidth = 800; // Adjust the width as needed
            const canvasHeight = canvasWidth / aspectRatio;

            canvas.setWidth(canvasWidth);
            canvas.setHeight(canvasHeight);

            canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas), {
              scaleX: canvasWidth / img.width,
              scaleY: canvasHeight / img.height,
            });
          },
          { crossOrigin: 'anonymous' }
        );
      }
    }
  }, [uploadedImage, canvas]);


///////////// get class images

const fetchClassImages = async (project_name, classNames) => {
    try {
      const sanitizedProjectName = project_name.replace(/\s+/g, '_');
      console.log('Fetching class images with project name:', sanitizedProjectName, 'and class names:', classNames);
      const response = await axios.post('http://127.0.0.1:5000/list-class-images', {
        project_name: sanitizedProjectName,
        class_names: classNames
      });

      if (response.data) {
        console.log('Received class images:', response.data);
        return response.data; // This will be an object with class names as keys and array of image URLs as values
      } else {
        throw new Error('No data received');
      }
    } catch (error) {
      console.error('Error fetching class images:', error);
      return {};
    }
  };

  // Example usage within a useEffect
  useEffect(() => {
    const fetchAndSetClassImages = async () => {
      const classImages = await fetchClassImages(project_name, classNames);
      console.log('Setting class images:', classImages);
      setClassImages(classImages); // Assuming you have a state variable named classImages
    };
  
    if (project_name && classNames.length > 0) {
      fetchAndSetClassImages();
    }
  }, [project_name, classNames]);



//////////////zoom///////////////////////////



  const [zoomLevel, setZoomLevel] = useState(1);

  const handleZoomIn = () => {
    if (canvas && zoomLevel < 3) { // Adjust the maximum zoom level as needed
      const newZoom = zoomLevel + 0.1; // Adjust the zoom increment as needed
      setZoomLevel(newZoom);
      canvas.setZoom(newZoom);
    }
  };

  const handleZoomOut = () => {
    if (canvas && zoomLevel > 0.5) { // Adjust the minimum zoom level as needed
      const newZoom = zoomLevel - 0.1; // Adjust the zoom decrement as needed
      setZoomLevel(newZoom);
      canvas.setZoom(newZoom);
    }
  };


  const handleResetZoom = () => {
    if (canvas) {
      canvas.setViewportTransform([1, 0, 0, 1, 0, 0]); // Reset zoom and pan
      setZoomLevel(1); // Reset zoom level state
    }
  };

  /////////////Panning///////////////

// Add a state variable to track whether an annotation is being moved or modified
const [annotationInProgress, setAnnotationInProgress] = useState(false);

// Function to handle annotation movement start
const handleAnnotationMovementStart = () => {
  setAnnotationInProgress(true);
};

// Function to handle annotation movement end
const handleAnnotationMovementEnd = () => {
  setAnnotationInProgress(false);
};

useEffect(() => {
  // Add event listeners to detect annotation movement start and end
  if (canvas) {
    canvas.on('object:moving', handleAnnotationMovementStart);
    canvas.on('object:scaling', handleAnnotationMovementStart);
    canvas.on('object:rotating', handleAnnotationMovementStart);
    canvas.on('object:modified', handleAnnotationMovementEnd);
  }

  // Remove event listeners on cleanup
  return () => {
    if (canvas) {
      canvas.off('object:moving', handleAnnotationMovementStart);
      canvas.off('object:scaling', handleAnnotationMovementStart);
      canvas.off('object:rotating', handleAnnotationMovementStart);
      canvas.off('object:modified', handleAnnotationMovementEnd);
    }
  };
}, [canvas]);

// Function to handle panning
const handlePan = (opt) => {
  if (!annotationInProgress && canvas) {
    const delta = opt.e.deltaY;
    const zoom = canvas.getZoom();
    let zoomFactor = 0.95; // Adjust the zoom factor

    if (delta > 0) {
      zoomFactor = 1.05;
    }

    const newZoom = zoom * zoomFactor;
    canvas.zoomToPoint({ x: opt.e.offsetX, y: opt.e.offsetY }, newZoom);
    opt.e.preventDefault();
    opt.e.stopPropagation();
  }
};

useEffect(() => {
  if (canvas) {
    // Add event listener for panning
    canvas.on('mouse:wheel', handlePan);

    return () => {
      // Remove event listener on cleanup
      canvas.off('mouse:wheel', handlePan);
    };
  }
}, [canvas, annotationInProgress]);

  
  // Adjusted to calculate relative coordinates
  const calculateRelativeCoordinates = (canvasPoint) => {
    const imageObj = canvas.backgroundImage;
    if (!imageObj) return canvasPoint; // Fallback if no background image

    const zoom = canvas.getZoom();
    const scaleX = imageObj.scaleX;
    const scaleY = imageObj.scaleY;
    const left = imageObj.left;
    const top = imageObj.top;

    const relativeX = (canvasPoint.x - left) / (zoom * scaleX);
    const relativeY = (canvasPoint.y - top) / (zoom * scaleY);
    return { x: relativeX, y: relativeY };
  };

  ////////////////////////////////////////
  const addPolygon = () => {
    if (canvas) {
      let points = [];
      let polygon = null;
      let isDrawing = true;
  
      const mouseDownHandler = (options) => {
        if (isDrawing) {
          const pointer = canvas.getPointer(options.e);
          points.push({ x: pointer.x, y: pointer.y });
  
          if (polygon) {
            polygon.set({ points: [...points] });
          } else {
            polygon = new fabric.Polygon(points, {
              stroke: annotationColor,
              strokeWidth: 1,
              fill: hexToRGBA(annotationColor, 0.3),
              selectable: false, // Set selectable to false initially
              objectCaching: false,
              perPixelTargetFind: true,
            });
            canvas.add(polygon);
          }
          canvas.renderAll();
        }
      };
  
      const mouseMoveHandler = (options) => {
        if (points.length && isDrawing) {
          const pointer = canvas.getPointer(options.e);
          const tempPoints = [...points, { x: pointer.x, y: pointer.y }];
          polygon.set({ points: tempPoints });
          canvas.renderAll();
        }
      };
  
      const doubleClickHandler = () => {
        isDrawing = false;
        canvas.off('mouse:down', mouseDownHandler);
        canvas.off('mouse:move', mouseMoveHandler);
        canvas.off('mouse:dblclick', doubleClickHandler);
      
        polygon.set({ selectable: true }); // Set selectable to true after drawing is finished
        
        // Convert polygon points to absolute coordinates if necessary
        const absolutePoints = polygon.points.map(p => ({
          x: p.x + polygon.left,
          y: p.y + polygon.top
        }));
        const id = `polygon-${Date.now()}-${Math.random()}`; // Generate a unique ID
        const annotation = { id, type: 'Polygon', label: labelText, points: absolutePoints };
      
        setAnnotations(prevAnnotations => [...prevAnnotations, annotation]);
      
        canvas.renderAll();
      };
      
  
      canvas.on('mouse:down', mouseDownHandler);
      canvas.on('mouse:move', mouseMoveHandler);
      canvas.on('mouse:dblclick', doubleClickHandler);
    }
  };
  


  //////////////////////////////////////

  /////////////////////////////////////
  const enableFreehandDrawing = () => {
    if (canvas) {
      canvas.isDrawingMode = true;
      canvas.freeDrawingBrush = new fabric.PencilBrush(canvas);
      updateFreeDrawingBrushColor(annotationColor); // Set initial color
  
      canvas.freeDrawingBrush.width = 1; // Set the brush width
    }
  };
  
  const updateFreeDrawingBrushColor = (color) => {
    if (canvas && canvas.freeDrawingBrush) {
      canvas.freeDrawingBrush.color = color;
      canvas.freeDrawingBrush.stroke = color; // Also update stroke color for the brush
    }
  };
  
  
  useEffect(() => {
    updateFreeDrawingBrushColor(annotationColor);
  }, [annotationColor, canvas]);
  

  useEffect(() => {
    if (canvas) {
      const handlePathCreated = (e) => {
        if (canvas.isDrawingMode) {
          const path = e.path;
          const pathData = path.path;
          const points = [];
          
          // Convert path data to points at regular intervals
          for (let i = 0; i < pathData.length; i += 5) {
            points.push({
              x: pathData[i][1],
              y: pathData[i][2]
            });
          }
          
          // Create a new polygon from the points
          const polygon = new fabric.Polygon(points, {
            fill: hexToRGBA(annotationColor, 0.3),
            stroke: annotationColor,
            strokeWidth: 1,
            selectable: true,
            objectCaching: false,
          });
  
          // Remove the freehand path
          canvas.remove(path);
          // Add the closed shape as a polygon for uniform handling
          canvas.add(polygon);
          canvas.isDrawingMode = false;
  
          // Store the annotation with its type and points
          const annotation = { type: 'Freehand', label: labelText, points: points };
          setAnnotations(prevAnnotations => [...prevAnnotations, annotation]);
          setLabelText('');
          canvas.renderAll();
        }
      };
  
      // Listen for path creation events
      canvas.on('path:created', handlePathCreated);
  
      const handlePathModified = (e) => {
        const path = e.target;
        const annotationIndex = annotations.findIndex(annot => annot.object === path);
      
        if (annotationIndex !== -1 && annotations[annotationIndex].type === 'Freehand') {
          const updatedAnnotations = [...annotations];
          const polygonPoints = path.path.map(p => ({ x: p[1], y: p[2] }));
      
          // Update the corresponding polygon's points
          updatedAnnotations[annotationIndex].object.set({ points: polygonPoints });
          updatedAnnotations[annotationIndex].points = polygonPoints;
          setAnnotations(updatedAnnotations);
        }
      };
      
      
  
      // Listen for path modification events
      canvas.on('path:modified', handlePathModified);
  
      return () => {
        // Cleanup by removing the event listener
        canvas.off('path:created', handlePathCreated);
        canvas.off('path:modified', handlePathModified);
      };
    }
  }, [canvas, labelText, annotationColor, annotations]);
  
  
  
  
  /////////////////////////////////////
  ////////////////////////////////////
  
  
 

  const saveAnnotatedImage = () => {
    if (canvas && selectedClass) {
      const imageData = canvas.toDataURL('image/png');
      const imageID = `image-${new Date().toISOString()}`;
      const sanitizedProjectName = project_name.replace(/\s+/g, '_'); // Replace spaces with underscores
  
      const annotationsData = annotations.map((annot, index) => ({
        imageID: imageID,
        patchID: `patch-${index}`,
        type: annot.type,
        label: annot.label,
        coordinates: formatCoordinatesForAnnotation(annot)
      }));
  
      axios.post('http://127.0.0.1:5000/save-annotated-image', {
        project_name: sanitizedProjectName,
        class_name: selectedClass,
        image_name: selectedImage.split('/').pop(),
        imageUrl: imageData,
        annotations: annotationsData
      })
      .then(response => {
        console.log(response.data.message);
        alert('Image and annotations saved successfully!');
       // Update the classImages state to include the new image
       setClassImages(prevClassImages => {
        const updatedClassImages = { ...prevClassImages };
        if (!updatedClassImages[selectedClass]) {
          updatedClassImages[selectedClass] = [];
        }
        // Add the new image URL to the class's image array
        const newImageUrl = `uploads/${sanitizedProjectName}_dataset/${selectedClass}/${selectedImage.split('/').pop()}`;
        updatedClassImages[selectedClass].push(newImageUrl);
        return updatedClassImages;
      });
    })
      .catch(error => {
        console.error('Error saving image:', error);
        alert(`Error saving image: ${error.response ? error.response.data.error : error.message}`);
      });
    } else {
      alert('Please select a class before saving.');
    }
  };


  
  ////////////////////////////////////
///// get class images ////
useEffect(() => {
  const fetchImages = async () => {
    try {
      const urls = images.map(image => ({
        id: image.id,
        filename: image.filename,
        filepath: `http://localhost:5000/${image.filepath}`
      }));
      setImageUrls(urls);
    } catch (error) {
      console.error("Error fetching images:", error);
    } finally {
      setIsLoading(false);
    }
  };

  if (images.length > 0) {
    fetchImages();
  }
}, [images]);

  const clearCanvas = () => {
    if (canvas) {
      canvas.clear();
      canvas.setBackgroundImage(null);
      setAnnotations([]);
    }
  };
  
  
  

  const handleImageSelection = (event) => {
    setSelectedImage(event.target.files[0]);
  };

  const handleImageUpload = () => {
    if (selectedImage) {
      // You can include the image upload logic here if needed
      setUploadedImage(URL.createObjectURL(selectedImage));
      setSelectedImage(null); // Reset selected image after upload
    }
  };

  const addRect = () => {
    if (canvas) {
      const rect = new fabric.Rect({
        left: 100,
        top: 100,
        fill: hexToRGBA(annotationColor, 0.3),
        stroke: annotationColor,
        strokeWidth: 0.2,
        width: 50,
        height: 50,
        selectable: true,
        
      });
  
      const annotation = { type: 'Rect', label: labelText, object: rect };
      
      canvas.add(rect); // Add only the rectangle to the canvas
      canvas.renderAll();
  
      setAnnotations(prevAnnotations => [...prevAnnotations, annotation]);
      setLabelText('');
    }
  };
  
  // Inside the addEllipse and addLine functions, modify as follows:

const addEllipse = () => {
  if (canvas) {
    const ellipse = new fabric.Ellipse({
      left: 150,
      top: 150,
      fill: hexToRGBA(annotationColor, 0.3),
      stroke: annotationColor,
      strokeWidth: 1,
      rx: 30,
      ry: 50,
      selectable: true,
    });

    const annotation = { type: 'Ellipse', label: labelText, object: ellipse };
    const text = new fabric.Text(labelText, {
      left: 150, // Adjust as needed
      top: 150, // Adjust as needed
      fontSize: 30,
      fill: 'black',
    });

    canvas.add(ellipse); 
    canvas.renderAll();

    setAnnotations(prevAnnotations => [...prevAnnotations, annotation]);
    setLabelText('');
  }
};

const addLine = () => {
  if (canvas) {
    // Coordinates for line start and end points
    const lineCoordinates = [50, 50, 200, 200];
    const line = new fabric.Line(lineCoordinates, {
      fill: 'transparent',
      stroke: annotationColor,
      strokeWidth: 1,
      selectable: true,
    });

    canvas.add(line);
    canvas.renderAll();

    const annotation = {
      type: 'Line',
      label: labelText,
      coordinates: [{x: line.x1, y: line.y1}, {x: line.x2, y: line.y2}], // Store start and end points
      object: line // Storing the fabric object may not be necessary depending on your implementation
    };

    setAnnotations(prevAnnotations => [...prevAnnotations, annotation]);
    setLabelText('');
  }
};



  const deleteSelected = () => {
    if (canvas) {
      const activeObject = canvas.getActiveObject();
      if (activeObject) {
        // Remove the associated label text
        const associatedText = annotations.find(annotation => annotation.object === activeObject)?.label;
        const textObj = canvas.getObjects('text').find(obj => obj.text === associatedText);
        if (textObj) {
          canvas.remove(textObj);
        }

        // Find the index of the annotation corresponding to the deleted object
        const deletedAnnotationIndex = annotations.findIndex(item => item.object === activeObject);

        if (deletedAnnotationIndex !== -1) {
          // Remove the annotation from the annotations array
          const updatedAnnotations = [...annotations];
          updatedAnnotations.splice(deletedAnnotationIndex, 1);
          setAnnotations(updatedAnnotations);
        } else {
          // Check if the active object is a freehand or polygon object
          if (activeObject.type === 'path' || activeObject.type === 'polygon') {
            // Search for the associated annotation by points
            const points = activeObject.type === 'path' ? activeObject.path : activeObject.points;
            const deletedAnnotationIndexByPoints = annotations.findIndex(item => {
              if (item.points.length !== points.length) return false;
              for (let i = 0; i < points.length; i++) {
                if (item.points[i].x !== points[i].x || item.points[i].y !== points[i].y) {
                  return false;
                }
              }
              return true;
            });

            // Remove the annotation if found
            if (deletedAnnotationIndexByPoints !== -1) {
              const updatedAnnotationsByPoints = [...annotations];
              updatedAnnotationsByPoints.splice(deletedAnnotationIndexByPoints, 1);
              setAnnotations(updatedAnnotationsByPoints);
            }
          }
        }

        // Remove the annotation object
        canvas.remove(activeObject);
        canvas.renderAll();
      }
    }
  };

  
  const clearAnnotations = () => {
    if (canvas) {
      canvas.getObjects().forEach(obj => {
        canvas.remove(obj);
      });
      canvas.renderAll();
      setAnnotations([]); // Clear all annotations
    }
  };
  

  useEffect(() => {
    let newCanvas = null; // Define the canvas variable outside useEffect

    const handleZoom = (opt) => {
      const delta = opt.e.deltaY;
      const zoom = newCanvas.getZoom();
      let zoomFactor = 0.95; // Adjust the zoom factor

      if (delta > 0) {
        zoomFactor = 1.05;
      }

      const newZoom = zoom * zoomFactor;
      newCanvas.zoomToPoint({ x: opt.e.offsetX, y: opt.e.offsetY }, newZoom);
      opt.e.preventDefault();
      opt.e.stopPropagation();
    };

    if (canvasRef.current && imageUrl) {
      newCanvas = new fabric.Canvas(canvasRef.current, {
        selection: true,
        backgroundColor: '#f0f0f0',
      });

      fabric.Image.fromURL(
        imageUrl,
        img => {
          const aspectRatio = img.width / img.height;
          const canvasWidth = 800; // Adjust the width as needed
          const canvasHeight = canvasWidth / aspectRatio;

          newCanvas.setWidth(canvasWidth);
          newCanvas.setHeight(canvasHeight);

          newCanvas.setBackgroundImage(img, newCanvas.renderAll.bind(newCanvas), {
            scaleX: canvasWidth / img.width,
            scaleY: canvasHeight / img.height,
          });
        },
        { crossOrigin: 'anonymous' }
      );

      newCanvas.off('mouse:wheel', handleZoom);

      setCanvas(newCanvas);
    }

    return () => {
      if (newCanvas) {
        newCanvas.off('mouse:wheel', handleZoom); // Remove the event listener on unmount
        newCanvas.dispose(); // Dispose the canvas to prevent memory leaks
        setCanvas(null);
      }
    };
  }, [imageUrl]);



  useEffect(() => {
    const handleObjectModified = (e) => {
      const updatedObject = e.target;
      const annotationIndex = annotations.findIndex(annot => annot.object === updatedObject);
    
      if (annotationIndex !== -1) {
        let updatedAnnotations = [...annotations];
        let annotation = updatedAnnotations[annotationIndex];
    
        // Update coordinates based on annotation type
        switch (annotation.type) {
          case 'Line':
            // Calculate the world coordinates of the line's start and end points
            const x1World = updatedObject.x1 + updatedObject.left;
            const y1World = updatedObject.y1 + updatedObject.top;
            const x2World = updatedObject.x2 + updatedObject.left;
            const y2World = updatedObject.y2 + updatedObject.top;
            // Update the annotation coordinates
            annotation.coordinates = [{ x: x1World, y: y1World }, { x: x2World, y: y2World }];
            break;
          case 'Ellipse':
            // Ellipse's center is at (left + rx, top + ry)
            annotation.coordinates = { cx: updatedObject.left + updatedObject.rx, cy: updatedObject.top + updatedObject.ry, rx: updatedObject.rx, ry: updatedObject.ry };
            break;
          case 'Polygon':
          case 'Freehand':
            // For polygons and freehand drawings, update the points based on the object's new position
            annotation.points = updatedObject.points.map(pt => ({
              x: pt.x + updatedObject.left,
              y: pt.y + updatedObject.top,
            }));
            break;
          // Add cases for other types as needed
        }
    
        updatedAnnotations[annotationIndex] = annotation; // Replace the old annotation with the updated one
        setAnnotations(updatedAnnotations); // Update the annotations state
      }
    };
    
  
    if (canvas) {
      canvas.on('object:modified', handleObjectModified);
    }
  
    return () => {
      if (canvas) {
        canvas.off('object:modified', handleObjectModified);
      }
    };
  }, [canvas, annotations]);
  
  // Adjusted formatCoordinatesForAnnotation function to handle coordinates for each annotation type

  function formatCoordinatesForAnnotation(annot) {
    let formattedCoordinates;
    switch (annot.type) {
      case 'Rect':
        // For rectangles, store the top-left corner, width, and height
        formattedCoordinates = {
          x: annot.object.left,
          y: annot.object.top,
          width: annot.object.width * annot.object.scaleX,
          height: annot.object.height * annot.object.scaleY
        };
        break;
      case 'Line':
        // Store the start and end points of the line
        formattedCoordinates = [
          { x: annot.coordinates[0].x, y: annot.coordinates[0].y },
          { x: annot.coordinates[1].x, y: annot.coordinates[1].y }
        ];
        break;
      case 'Ellipse':
        // Store the center and radii of the ellipse
        formattedCoordinates = {
          cx: annot.coordinates.cx,
          cy: annot.coordinates.cy,
          rx: annot.coordinates.rx,
          ry: annot.coordinates.ry
        };
        break;
      case 'Polygon':
      case 'Freehand':
        // Store the points of the polygon or freehand drawing
        formattedCoordinates = annot.points.map(pt => ({
          x: pt.x,
          y: pt.y,
        }));
        break;
      default:
        formattedCoordinates = [];
    }
    return formattedCoordinates;
  }
  




  
  useEffect(() => {
    if (canvas) {
      canvas.getObjects('text').forEach(text => canvas.remove(text)); // Clear all existing text objects (labels)
  
      annotations.forEach(annotation => {
        const { object, label } = annotation;
        if (object && label) {
          let textLeft = 0;
          let textTop = 0;
  
          if (object.type === 'Rect') {
            textLeft = object.left + 5;
            textTop = object.top + 5;
          } else if (object.type === 'Ellipse') {
            textLeft = object.left + object.rx / 2;
            textTop = object.top + object.ry / 2;
          } else if (object.type === 'Line') {
            textLeft = (object.x1 + object.x2) / 2;
            textTop = (object.y1 + object.y2) / 2;
          }
  
          const text = new fabric.Text(label, {
            left: textLeft,
            top: textTop,
            fontSize: 16,
            fill: 'black',
          });
  
          //canvas.add(text);
        }
      });
  
      canvas.renderAll();
    }
  }, [annotations, canvas]);

  const downloadAnnotations = () => {
    let csvContent = "data:text/csv;charset=utf-8,";
    csvContent += "Patch ID,Type,Label,Coordinates\r\n"; // Header row
  
    annotations.forEach((annotation, index) => {
      const patchID = index + 1; // Start Patch IDs from 1
      const { type, label } = annotation;
      let coordinates = '';
  
      switch (type) {
        case 'Polygon':
        case 'Freehand':
          coordinates = annotation.points.map(p => `(${p.x.toFixed(2)}, ${p.y.toFixed(2)})`).join(', ');
          break;
        case 'Rect':
          const coords = [
            { x: annotation.object.left, y: annotation.object.top },
            { x: annotation.object.left + annotation.object.width * annotation.object.scaleX, y: annotation.object.top },
            { x: annotation.object.left, y: annotation.object.top + annotation.object.height * annotation.object.scaleY },
            { x: annotation.object.left + annotation.object.width * annotation.object.scaleX, y: annotation.object.top + annotation.object.height * annotation.object.scaleY },
          ];
          coordinates = coords.map(coord => `(${coord.x.toFixed(2)}, ${coord.y.toFixed(2)})`).join(', ');
          break;
        case 'Line':
          coordinates = `(${annotation.coordinates[0].x.toFixed(2)}, ${annotation.coordinates[0].y.toFixed(2)}) to (${annotation.coordinates[1].x.toFixed(2)}, ${annotation.coordinates[1].y.toFixed(2)})`;
          break;
        case 'Ellipse':
          const { cx, cy, rx, ry } = annotation.coordinates;
          coordinates = `Center: (${cx.toFixed(2)}, ${cy.toFixed(2)}), RX: ${rx.toFixed(2)}, RY: ${ry.toFixed(2)}`;
          break;
        default:
          coordinates = 'N/A';
      }
  
      let row = `${patchID},${type},${label},"${coordinates}"\r\n`;
      csvContent += row;
    });
  
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', 'annotations.csv');
    document.body.appendChild(link); // Required for FF
  
    link.click(); // This will download the file
    document.body.removeChild(link); // Clean up
  };
  

  return (
    <div style={{ paddingLeft: '40px', paddingRight: '100px' }}> {/* Adjusted left and right side spacing */}
      <Header /> 
      
      <h1 style={{ fontSize: '32px', fontWeight: 'bold', marginBottom: '20px' }}>Tissue Annotation</h1> {/* Bigger and bold heading */}
      <Link 
  to="/CreateProject" 
  style={{ 
    display: 'inline-block',
    width: '60px', // Adjusted width
    padding: '5px',
    marginBottom:'10px',
    marginLeft:'10px',
    border: '1px solid #ccc', // Grey border
    borderRadius: '4px',
    marginTop: '-10px',
    textDecoration: 'none', // Remove underline from the link
    backgroundColor: '#f0f0f0', // Button background color
    color: '#000', // Button text color
    textAlign: 'center' // Center text
  }}
>
  Back
</Link>
      <div><TissueCarousel images={imageUrls} onSelect={handleImageSelect}/></div>
      <div style={{ display: 'flex' }}>
        <div style={{ flex: '1', marginRight: '20px', marginBottom: '20px', marginTop: '20px' }}>
          
          

          {imageUrl && (
            <div>
              
              <canvas
                ref={canvasRef}
                style={{ border: '1px solid #000', maxWidth: '100%' }}
              />
            </div>
          )}
          
      
        </div>

        <div style={{ flex: '1', display: 'flex', flexDirection: 'column', marginTop: '80px' }}>
          <input
            type="text"
            value={labelText}
            onChange={(e) => setLabelText(e.target.value)}
            placeholder="Enter label text"
            style={{ 
              width: '150px', // Adjusted width
              padding: '8px',
              margin: '8px 0',
              border: '1px solid #ccc', // Grey border
              borderRadius: '4px',
              marginTop: '-60px'
            }}
          />
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <button
              onClick={addRect}
              style={{
                padding: '8px 16px',
                margin: '8px 0',
                backgroundColor: '#6415FF', // Purple color
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                width: '150px', // Adjusted width
              }}
            >
              Add Rectangle
            </button>

            {/* <button onClick={addPolygon}
            style={{
              padding: '8px 16px',
              margin: '8px 0',
              backgroundColor: '#6415FF', // Purple color
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              width: '150px',
            }} // Adjusted width
              >
                Add Polygon
                </button> */}
                <button onClick={enableFreehandDrawing}
                style={{
                  padding: '8px 16px',
              margin: '8px 0',
              backgroundColor: '#6415FF', // Purple color
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              width: '150px',
            }}>
            Freehand Draw
            </button>

            <button
              onClick={addLine}
              style={{
                padding: '8px 16px',
                margin: '8px 0',
                backgroundColor: '#6415FF', // Purple color
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                width: '150px', // Adjusted width
              }}
            >
              Add Line
            </button>
            
            {/* <button
              onClick={clearCanvas}
              style={{
                padding: '8px 16px',
                margin: '8px 0',
                backgroundColor: '#6415FF', // Purple color
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                width: '150px', // Adjusted width
              }}
            >
              Clear Canvas
            </button> */}

            
            

          </div>
          {/* Color Picker */}
      <input 
        type="color" 
        value={annotationColor} 
        onChange={(e) => setAnnotationColor(e.target.value)}
        style={{ width: '150px', margin: '0px' }}
      />
      <button
          onClick={deleteSelected}
          style={{
            padding: '8px 16px',
            margin: '8px 0',
            backgroundColor: '#6415FF', // Purple color
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            width: '150px', // Adjusted width
          }}
        >
          Delete Selected
        </button>

        <button
              onClick={clearAnnotations}
              style={{
                padding: '8px 16px',
                margin: '8px 0',
                backgroundColor: '#6415FF', // Purple color
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                width: '150px', // Adjusted width
              }}
            >
              Clear
            </button>
            <div>
            <button
              onClick={saveAnnotatedImage}
              style={{
                padding: '8px 16px',
                margin: '8px 0',
                backgroundColor: '#6415FF', // Purple color
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                width: '70px', // Adjusted width
                height: '40px',
                marginRight: '10px',
                paddingLeft:'24px'
            }}
          >
          <svg xmlns="http://www.w3.org/2000/svg" enable-background="new 0 0 92 92" viewBox="0 0 150 150" id="Save"><switch><g fill="#ffffff" class="color000000 svgShape"><path d="M5273.1,2400.1v-2c0-2.8-5-4-9.7-4s-9.7,1.3-9.7,4v2c0,1.8,0.7,3.6,2,4.9l5,4.9c0.3,0.3,0.4,0.6,0.4,1v6.4
				c0,0.4,0.2,0.7,0.6,0.8l2.9,0.9c0.5,0.1,1-0.2,1-0.8v-7.2c0-0.4,0.2-0.7,0.4-1l5.1-5C5272.4,2403.7,5273.1,2401.9,5273.1,2400.1z
				 M5263.4,2400c-4.8,0-7.4-1.3-7.5-1.8v0c0.1-0.5,2.7-1.8,7.5-1.8c4.8,0,7.3,1.3,7.5,1.8C5270.7,2398.7,5268.2,2400,5263.4,2400z" fill="#ffffff" class="color000000 svgShape"></path><path d="M5268.4 2410.3c-.6 0-1 .4-1 1 0 .6.4 1 1 1h4.3c.6 0 1-.4 1-1 0-.6-.4-1-1-1H5268.4zM5272.7 2413.7h-4.3c-.6 0-1 .4-1 1 0 .6.4 1 1 1h4.3c.6 0 1-.4 1-1C5273.7 2414.1 5273.3 2413.7 5272.7 2413.7zM5272.7 2417h-4.3c-.6 0-1 .4-1 1 0 .6.4 1 1 1h4.3c.6 0 1-.4 1-1C5273.7 2417.5 5273.3 2417 5272.7 2417z" fill="#ffffff" class="color000000 svgShape"></path><g fill="#ffffff" class="color000000 svgShape"><path d="M94.6,25.9L73.7,5c-1.6-1.6-3.8-2.5-6-2.5H58V20c0,1.8-1.5,3.3-3.3,3.3H27.3c-1.8,0-3.3-1.5-3.3-3.3V2.5H11.4
				c-4.7,0-8.5,3.8-8.5,8.5v78c0,4.7,3.8,8.5,8.5,8.5h77.3c4.7,0,8.5-3.8,8.5-8.5V31.9C97.1,29.7,96.2,27.5,94.6,25.9z M76.9,78.2
				c0,1.8-1.5,3.3-3.3,3.3H26.3c-1.8,0-3.3-1.5-3.3-3.3V45.4c0-1.8,1.5-3.3,3.3-3.3h47.4c1.8,0,3.3,1.5,3.3,3.3V78.2z" fill="#ffffff" class="color000000 svgShape"></path><path d="M44.2 17.7h6.4c.7 0 1.3-.6 1.3-1.3V3.8c0-.7-.6-1.3-1.3-1.3h-6.4c-.7 0-1.3.6-1.3 1.3v12.5C42.9 17.1 43.5 17.7 44.2 17.7zM63.9 51.3H36.1c-1.9 0-3.5 1.5-3.5 3.5 0 1.9 1.5 3.5 3.5 3.5h27.7c1.9 0 3.5-1.5 3.5-3.5C67.3 52.8 65.8 51.3 63.9 51.3zM63.9 65.5H36.1c-1.9 0-3.5 1.5-3.5 3.5 0 1.9 1.5 3.5 3.5 3.5h27.7c1.9 0 3.5-1.5 3.5-3.5C67.3 67.1 65.8 65.5 63.9 65.5z" fill="#ffffff" class="color000000 svgShape"></path></g></g></switch></svg>
          </button>

          <button onClick={downloadAnnotations} style={{
                padding: '8px 16px',
                margin: '8px 0',
                backgroundColor: '#6415FF', // Purple color
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                width: '70px', // Adjusted width
                height: '40px',
                paddingLeft:'24px'
}}>
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2300 2300" id="Download"><path d="M1344 1344q0-26-19-45t-45-19-45 19-19 45 19 45 45 19 45-19 19-45zm256 0q0-26-19-45t-45-19-45 19-19 45 19 45 45 19 45-19 19-45zm128-224v320q0 40-28 68t-68 28H160q-40 0-68-28t-28-68v-320q0-40 28-68t68-28h465l135 136q58 56 136 56t136-56l136-136h464q40 0 68 28t28 68zm-325-569q17 41-14 70l-448 448q-18 19-45 19t-45-19L403 621q-31-29-14-70 17-39 59-39h256V64q0-26 19-45t45-19h256q26 0 45 19t19 45v448h256q42 0 59 39z" fill="#ffffff" class="color000000 svgShape"></path></svg>
</button>
</div>
<div>
        {/* Add zoom in and zoom out buttons */}
        <button
          onClick={handleZoomIn}
          style={{
            padding: '8px 8px',
            margin: '8px 0',
            backgroundColor: '#6415FF',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            width: '40px',
            height: '40px',
            marginRight: '12px',
          }}
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 92 92" id="Zoom-in">
        <path d="M90.7 84.3 55.9 50c4.2-5.3 6.6-11.9 6.6-19 0-17.1-14-31-31.3-31S0 13.9 0 31s14 31 31.3 31c6.8 0 13-2.1 18.1-5.8l35 34.5c.9.9 2 1.3 3.1 1.3 1.2 0 2.3-.4 3.2-1.3 1.7-1.8 1.7-4.6 0-6.4zM8 31C8 18.3 18.4 8 31.3 8s23.3 10.3 23.3 23-10.5 23-23.3 23S8 43.6 8 31zm38.9.5c0 1.9-1.6 3.5-3.5 3.5H35v8c0 1.9-1.6 3.5-3.5 3.5S28 44.9 28 43v-8h-8.9c-1.9 0-3.5-1.6-3.5-3.5s1.6-3.5 3.5-3.5H28v-9.1c0-1.9 1.6-3.5 3.5-3.5S35 17 35 18.9V28h8.4c2 0 3.5 1.6 3.5 3.5z" fill="#ffffff" class="color000000 svgShape"></path>
    </svg>
        </button>
        <button
          onClick={handleZoomOut}
          style={{
            padding: '8px 8px',
            margin: '8px 0',
            backgroundColor: '#6415FF',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            width: '40px',
            height: '40px',
            marginRight: '9px'
          }}
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" id="ZoomOut"><circle cx="22.01" cy="22" r="20.02" fill="none" stroke="#ffffff" stroke-miterlimit="10" stroke-width="4" class="colorStroke010101 svgStroke"></circle><line x1="12.01" x2="32.01" y1="22" y2="22" fill="none" stroke="#ffffff" stroke-miterlimit="10" stroke-width="4" class="colorStroke010101 svgStroke"></line><path fill="none" stroke="#ffffff" stroke-miterlimit="10" stroke-width="4" d="M35.94,36,62,62" class="colorStroke010101 svgStroke"></path></svg>
        </button>
        <button
          onClick={handleResetZoom}
          style={{
          padding: '8px 8px',
          margin: '8px 8px',
          backgroundColor: '#6415FF', // Purple color
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer',
          width: '40px',
          height: '40px',
          }}
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40" id="search"><path fill="#ffffff" d="M20.745 32.62c2.883 0 5.606-1.022 7.773-2.881L39.052 40.3a.996.996 0 0 0 1.414.002 1 1 0 0 0 .002-1.414L29.925 28.319c3.947-4.714 3.717-11.773-.705-16.205-2.264-2.27-5.274-3.52-8.476-3.52s-6.212 1.25-8.476 3.52c-4.671 4.683-4.671 12.304 0 16.987a11.9 11.9 0 0 0 8.477 3.519zm-7.06-19.094c1.886-1.891 4.393-2.932 7.06-2.932s5.174 1.041 7.06 2.932c3.895 3.905 3.895 10.258 0 14.163-1.886 1.891-4.393 2.932-7.06 2.932s-5.174-1.041-7.06-2.932c-3.894-3.905-3.894-10.258 0-14.163z" class="color231f20 svgShape"></path></svg>
        </button>
        


      </div>
      {/* Dropdown for class selection */}
      <select 
            value={selectedClass} 
            onChange={(e) => setSelectedClass(e.target.value)} 
            style={{ width: '150px', padding: '8px', margin: '8px 0' }}>
            <option value="" disabled>Select Class</option>
            {classNames.map((className, index) => (
              <option key={index} value={className}>{className}</option>
            ))}
          </select>

        </div>
      </div>

      <div style={{ position: 'absolute', right: '0px', top: '210px', width: '480px', maxHeight: '800px', overflowY: 'auto', border: '1px solid #ccc', padding: '4px', borderRadius: '5px', backgroundColor: '#f9f9f9', zIndex: '1', marginRight:'10px' }}>
  <table style={{ width: '100%', borderCollapse: 'collapse', tableLayout: 'fixed' }}>
    <thead>
      <tr>
        <th style={{ border: '1px solid #ccc', padding: '4px', color: '#ffff', borderRight: 'none', borderColor: '#6415FF',  backgroundColor: '#6415FF', width: '90px' }}>Patch ID</th>
        <th style={{ border: '1px solid #ccc', padding: '4px', color: '#ffff',borderRight: 'none', borderColor: '#6415FF', backgroundColor: '#6415FF', width: '120px' }}>Type</th>
        <th style={{ border: '1px solid #ccc', padding: '8px', color: '#ffff',borderRight: 'none', borderColor: '#6415FF', backgroundColor: '#6415FF', width: '120px' }}>Label</th>
        <th style={{ border: '1px solid #ccc', padding: '8px', color: '#ffff',borderColor: '#6415FF', backgroundColor: '#6415FF', width: '120px' }}>Coordinates</th>
      </tr>
    </thead>
  </table>
  <div style={{ overflowY: 'auto', maxHeight: '400px', scrollbarWidth:'thin' }}> {/* Adjust the maxHeight accordingly to fit within your container div */}
    <table style={{ width: '100%', borderCollapse: 'collapse', tableLayout: 'fixed' }}>
      <tbody>
        {annotations.map((annotation, index) => (
          <tr key={index}>
            <td style={{ border: '1px solid #ccc', padding: '4px', width: '88px', textAlign: 'center' }}>{index}</td>
            <td style={{ border: '1px solid #ccc', padding: '4px', width: '120px',textAlign: 'center' }}>{annotation.type}</td>
            <td style={{ border: '1px solid #ccc', padding: '8px', width: '120px' ,whiteSpace: 'nowrap', // Keeps the text on a single line
            overflow: 'hidden', // Hides any content that overflows the element's box
            textOverflow: 'ellipsis',textAlign: 'center'}}>{annotation.label}</td>
            <td style={{ border: '1px solid #ccc', padding: '8px', width: '120px',textAlign: 'center' }}>
              {/* Determine how to display coordinates based on the annotation type */}
              {(() => {
                switch (annotation.type) {
                  case 'Polygon':
                  case 'Freehand':
                    return annotation.points.map(p => `(${p.x.toFixed(2)}, ${p.y.toFixed(2)})`).join(', ');
                  case 'Rect':
                    const coords = [
                      { x: annotation.object.left, y: annotation.object.top },
                      { x: annotation.object.left + annotation.object.width * annotation.object.scaleX, y: annotation.object.top },
                      { x: annotation.object.left, y: annotation.object.top + annotation.object.height * annotation.object.scaleY },
                      { x: annotation.object.left + annotation.object.width * annotation.object.scaleX, y: annotation.object.top + annotation.object.height * annotation.object.scaleY },
                    ];
                    return coords.map(coord => `(${coord.x.toFixed(2)}, ${coord.y.toFixed(2)})`).join(', ');
                  case 'Line':
                    return `(${annotation.coordinates[0].x.toFixed(2)}, ${annotation.coordinates[0].y.toFixed(2)}) to (${annotation.coordinates[1].x.toFixed(2)}, ${annotation.coordinates[1].y.toFixed(2)})`;
                  case 'Ellipse':
                    const { cx, cy, rx, ry } = annotation.coordinates;
                    return `Center: (${cx.toFixed(2)}, ${cy.toFixed(2)}), RX: ${rx.toFixed(2)}, RY: ${ry.toFixed(2)}`;
                  default:
                    return 'N/A';
                }
              })()}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
</div>

{Object.keys(classImages).length > 0 && Object.keys(classImages).map(className => (
  <div key={className} style={{ marginTop: '20px' }}>
    <h3>{className}</h3>
    <AnnotatedImagesDisplay images={classImages[className].map(url => `http://localhost:5000/${url}`)} />
  </div>
))}





    </div>
  );
};

export default App;