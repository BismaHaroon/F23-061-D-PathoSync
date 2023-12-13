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
              selectable: false,
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
  
        polygon.set({ selectable: true });

        const annotation = { type: 'Polygon', label: labelText, object: polygon };
        setAnnotations(prevAnnotations => [...prevAnnotations, annotation]);
        polygon.set({ selectable: true, movable: true });
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
          const points = path.path.map(p => ({ x: p[1], y: p[2] }));
  
          // Close the path to form a polygon
          const polygon = new fabric.Polygon(points, {
            fill: hexToRGBA(annotationColor, 0.3), // Shaded fill color
            stroke: annotationColor,
            strokeWidth: 1,
            selectable: true,
            movable: true,
          });
  
          canvas.remove(path); // Remove the freehand path
          canvas.add(polygon); // Add the closed shape
          canvas.isDrawingMode = false; // Exit drawing mode
  
          // Capture the current label text
          const currentLabel = labelText;
        
        // Reset the labelText state if necessary
         setLabelText('');

          const annotation = { type: 'Freehand', label: labelText, object: polygon };
          setAnnotations(prevAnnotations => [...prevAnnotations, annotation]);
          setLabelText('');
          canvas.renderAll();
        }
      };
  
      canvas.on('path:created', handlePathCreated);
  
      return () => {
        canvas.off('path:created', handlePathCreated);
      };
    }
  }, [canvas, labelText]);
  
  
  /////////////////////////////////////
  ////////////////////////////////////

  const saveAnnotatedImage = () => {
    if (canvas) {
      const imageData = canvas.toDataURL('image/png'); // Convert canvas data to base64
  
      // Generate a unique image ID using the current datetime
      const imageID = `image-${new Date().toISOString()}`;
  
      const annotationsData = annotations.map((annot, index) => ({
        imageID: imageID,
        patchID: `patch-${index}`, // Assuming each annotation is given a sequential patch ID
        type: annot.type,
        label: annot.label
        // coordinates: annot.coordinates // include this if you manage to get coordinates working
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
        strokeWidth: 1,
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
    const line = new fabric.Line([50, 50, 200, 200], {
      fill: 'transparent',
      stroke: annotationColor,
      strokeWidth: 1,
      selectable: true,
    });

    const annotation = { type: 'Line', label: labelText, object: line };
    const text = new fabric.Text(labelText, {
      left: 50, // Adjust as needed
      top: 50, // Adjust as needed
      fontSize: 30,
      fill: 'black',
    });

    canvas.add(line);
    canvas.renderAll();

    setAnnotations(prevAnnotations => [...prevAnnotations, annotation]);
    setLabelText('');
  }
};

  

  // Add other annotation functions (dots, lines, polygons) similarly

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
  
        // Update the annotations state
        const filteredAnnotations = annotations.filter(item => item.object !== activeObject);
        setAnnotations(filteredAnnotations);
  
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

            <button onClick={addPolygon}
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
                </button>
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
              onClick={addEllipse}
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
              Add Ellipse
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
        </div>
      </div>

      <div style={{ position: 'absolute', right: '100px', top: '175px', zIndex: '1' }}>
  {/* Annotations positioned on the right */}
  {annotations.length > 0 ? (
    <div>
      <h3>Annotation Labels:</h3>
      <ul>
        {annotations.map((annotation, index) => (
          <li key={index}>
            Patch ID: {index}, Type: {annotation.type}, Label: {annotation.label}
          </li>
        ))}
      </ul>
    </div>
  ) : (
    <p>No annotations added yet.</p>
  )}
</div>
    </div>
  );
};

export default App;