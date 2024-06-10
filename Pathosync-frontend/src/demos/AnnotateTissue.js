import React, { useState, useEffect, useRef } from 'react';
import { fabric } from 'fabric';
import Header from "components/headers/light";
import axios from 'axios';
import { useParams } from "react-router-dom";
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


  console.log('processedImage:', processedImage);
  const [imageUrl, setImageUrl] = useState(null);
  useEffect(() => {
    setImageUrl(decodeURIComponent(processedImage));
  }, [processedImage]);

//////////////zoom///////////////////////////




///////////////////////////////////////////////////


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
    if (canvas) {
      const imageData = canvas.toDataURL('image/png');
      const imageID = `image-${new Date().toISOString()}`;
  
      const annotationsData = annotations.map((annot, index) => ({
        imageID: imageID,
        patchID: `patch-${index}`,
        type: annot.type,
        label: annot.label,
        // Handle coordinate formatting based on type here
        coordinates: formatCoordinatesForAnnotation(annot) // You'll need to implement this based on your data structure
      }));
  
      axios.post('http://127.0.0.1:5000/save-annotated-image', {
        imageUrl: imageData,
        annotations: annotationsData
      })
      .then(response => {
        console.log(response.data.message);
      })
      .catch(error => {
        console.error('Error saving image:', error);
      });
    }
  };
  
  
  
  ////////////////////////////////////


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
      <h1 style={{ fontSize: '32px', fontWeight: 'bold' }}>Tissue Annotation</h1> {/* Bigger and bold heading */}
      <div style={{ display: 'flex' }}>
        <div style={{ flex: '1', marginRight: '20px' }}>
          
          

          {imageUrl && (
            <div>
              <h2>Uploaded Image:</h2>
              <canvas
                ref={canvasRef}
                style={{ border: '1px solid #000', maxWidth: '100%' }}
              />
            </div>
          )}
          <div>
        {/* Add zoom in and zoom out buttons */}
        <button
          onClick={handleZoomIn}
          style={{
            padding: '8px 16px',
            margin: '8px 0',
            backgroundColor: '#6415FF',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            width: '120px',
            marginRight: '10px',
          }}
        >
          Zoom In
        </button>
        <button
          onClick={handleZoomOut}
          style={{
            padding: '8px 16px',
            margin: '8px 0',
            backgroundColor: '#6415FF',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            width: '120px',
          }}
        >
          Zoom Out
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
          width: '150px', // Adjusted width
          }}
        >
          Reset Zoom
        </button>
      </div>
      
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
              Clear Annotations
            </button>
            <button
              onClick={saveAnnotatedImage}
              style={{
              padding: '8px 16px',
              margin: '8px 0',
              backgroundColor: '#6415FF',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              width: '150px',
            }}
          >
          Save Image
          </button>

          </div>
          {/* Color Picker */}
      <input 
        type="color" 
        value={annotationColor} 
        onChange={(e) => setAnnotationColor(e.target.value)}
        style={{ width: '150px', margin: '0px' }}
      />
      <button onClick={downloadAnnotations} style={{
    padding: '8px 16px',
    margin: '8px 0',
    backgroundColor: '#6415FF', // Purple color
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    width: '150px',
}}>
    Download Annotations
</button>

        </div>
      </div>

      <div style={{ position: 'absolute', right: '0px', top: '175px', width: '480px', maxHeight: '400px', overflowY: 'auto', border: '1px solid #ccc', padding: '10px', borderRadius: '5px', backgroundColor: '#f9f9f9', zIndex: '1' }}>
  {annotations.length > 0 ? (
    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
      <thead>
        <tr>
          <th style={{ border: '1px solid #ccc', padding: '8px', backgroundColor: '#f2f2f2' }}>Patch ID</th>
          <th style={{ border: '1px solid #ccc', padding: '8px', backgroundColor: '#f2f2f2' }}>Type</th>
          <th style={{ border: '1px solid #ccc', padding: '8px', backgroundColor: '#f2f2f2' }}>Label</th>
          <th style={{ border: '1px solid #ccc', padding: '8px', backgroundColor: '#f2f2f2' }}>Coordinates</th>
        </tr>
      </thead>
      <tbody>
        {annotations.map((annotation, index) => (
          <tr key={index}>
            <td style={{ border: '1px solid #ccc', padding: '8px' }}>{index}</td>
            <td style={{ border: '1px solid #ccc', padding: '8px' }}>{annotation.type}</td>
            <td style={{ border: '1px solid #ccc', padding: '8px' }}>{annotation.label}</td>
            <td style={{ border: '1px solid #ccc', padding: '8px' }}>
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
  ) : (
    <p>No annotations added yet.</p>
  )}
</div>


    </div>
  );
};

export default App;