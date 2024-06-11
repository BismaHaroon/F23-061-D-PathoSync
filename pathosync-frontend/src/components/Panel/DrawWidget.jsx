import React, { useRef, useEffect, useState } from 'react';
import { fabric } from 'fabric';
import './AnnotationCanvas.css';

const AnnotationCanvas = () => {
  const canvasRef = useRef(null);
  const [canvas, setCanvas] = useState(null);

  useEffect(() => {
    const initCanvas = new fabric.Canvas(canvasRef.current, {
      isDrawingMode: false
    });
    setCanvas(initCanvas);

    // Zooming and panning
    initCanvas.on('mouse:wheel', function (opt) {
      let delta = opt.e.deltaY;
      let zoom = initCanvas.getZoom();
      zoom = zoom + delta / 200;
      if (zoom > 20) zoom = 20;
      if (zoom < 0.01) zoom = 0.01;
      initCanvas.zoomToPoint({ x: opt.e.offsetX, y: opt.e.offsetY }, zoom);
      opt.e.preventDefault();
      opt.e.stopPropagation();
    });
  }, []);

  return (
    <canvas ref={canvasRef} />
  );
};

const DrawWidget = () => {
  const [drawingType, setDrawingType] = useState('point');
  const [opts, setOpts] = useState({
    brush_shape: 'square',
    brush_size: 10,
    brush_screen: false,
    fixed_width: 100,
    fixed_height: 100
  });
  const [groups, setGroups] = useState([]);
  const [style, setStyle] = useState(null);

  useEffect(() => {
    // Fetch groups and style
    // ...

    // Set initial style
    setStyle(groups[0].id);
  }, []);

  const handleDrawingTypeChange = (type) => {
    setDrawingType(type);
  };

  const handleBrushShapeChange = (shape) => {
    setOpts({
      ...opts,
      brush_shape: shape
    });
  };

  const handleBrushSizeChange = (size) => {
    setOpts({
      ...opts,
      brush_size: size
    });
  };

  const handleBrushScreenChange = (screen) => {
    setOpts({
      ...opts,
      brush_screen: screen
    });
  };

  const handleFixedWidthChange = (width) => {
    setOpts({
      ...opts,
      fixed_width: width
    });
  };

  const handleFixedHeightChange = (height) => {
    setOpts({
      ...opts,
      fixed_height: height
    });
  };

  const handleGroupChange = (groupId) => {
    setStyle(groupId);
  };

  return (
    <div className="h-draw-widget">
      <div className="input-group input-group-sm h-style-group-row">
        <select className="form-control h-style-group" value={style} onChange={(e) => handleGroupChange(e.target.value)}>
          {groups.sortBy('id').map((group) => (
            <option key={group.id} value={group.id}>{group.id}</option>
          ))}
        </select>
        <div className="input-group-btn">
          <button className="btn btn-default h-configure-style-group" type="button">
            <span className="icon-cog" title="Configure style group. Keyboard shortcuts: &quot;q&quot; and &quot;w&quot; to select next and previous style group, respectively"></span>
          </button>
        </div>
      </div>
      <div className="btn-group btn-justified btn-group-sm h-draw-tools">
        <div className="btn-group btn-group-sm">
          <button className={`h-draw btn btn-default ${drawingType === 'point' ? 'active' : ''}`} type="button" data-type="point" title="Draw a new point (keyboard shortcut: o)" onClick={() => handleDrawingTypeChange('point')}>
            <span className="icon-circle"></span>Point
          </button>
        </div>
        <div className="btn-group btn-group-sm">
          <button className={`h-draw btn btn-default ${drawingType === 'rectangle' ? 'active' : ''}`} type="button" data-type="rectangle" title="Draw a new rectangle (keyboard shortcut: r)" onClick={() => handleDrawingTypeChange('rectangle')}>
            <span className="icon-check-empty"></span>Rectangle
          </button>
          <button className="btn btn-default h-dropdown-title h-brush-dropdown" type="button" data-target="#h-fixed-shape-controls">
            <i className="icon-down-open"></i>
          </button>
          <div className="h-fixed-shape-controls collapse">
            <div className="form-group h-fixed-shape-form-group input-sm">
              <label className="radio">
                <input className="h-fixed-shape" type="radio" name="h-fixed-shape" mode="unconstrained" />
                Unconstrained
              </label>
              <label className="radio">
                <input className="h-fixed-shape" type="radio" name="h-fixed-shape" mode="fixed_aspect_ratio" />
                Fixed Aspect Ratio
              </label>
              <label className="radio">
                <input className="h-fixed-shape" type="radio" name="h-fixed-shape" mode="fixed_size" />
                Fixed Size
              </label>
            </div>
            <div className="form-group h-fixed-shape-form-group h-fixed-values input-sm">
              <label>
                Width
                <input className="h-fixed-width" type="number" min="1" value={opts.fixed_width} onChange={(e) => handleFixedWidthChange(e.target.value)} />
              </label>
              <label>
                Height
                <input className="h-fixed-height" type="number" min="1" value={opts.fixed_height} onChange={(e) => handleFixedHeightChange(e.target.value)} />
              </label>
            </div>
          </div>
        </div>
        <div className="btn-group btn-group-sm">
          <button className={`h-draw btn btn-default ${drawingType === 'ellipse' ? 'active' : ''}`} type="button" data-type="ellipse" title="Draw a new ellipse (keyboard shortcut: i)" onClick={() => handleDrawingTypeChange('ellipse')}>
            <span className="icon-circle-empty flattenicon"></span>Ellipse
          </button>
          <button className="btn btn-default h-dropdown-title h-brush-dropdown" type="button" data-target="#h-fixed-shape-controls">
            <i className="icon-down-open"></i>
          </button>
          <div className="h-fixed-shape-controls collapse">
            <div className="form-group h-fixed-shape-form-group input-sm">
              <label className="radio">
                <input className="h-fixed-shape" type="radio" name="h-fixed-shape" mode="unconstrained" />
                Unconstrained
              </label>
              <label className="radio">
                <input className="h-fixed-shape" type="radio" name="h-fixed-shape" mode="fixed_aspect_ratio" />
                Fixed Aspect Ratio
              </label>
              <label className="radio">
                <input className="h-fixed-shape" type="radio" name="h-fixed-shape" mode="fixed_size" />
                Fixed Size
              </label>
            </div>
            <div className="form-group h-fixed-shape-form-group h-fixed-values input-sm">
              <label>
                Width
                <input className="h-fixed-width" type="number" min="1" value={opts.fixed_width} onChange={(e) => handleFixedWidthChange(e.target.value)} />
              </label>
              <label>
                Height
                <input className="h-fixed-height" type="number" min="1" value={opts.fixed_height} onChange={(e) => handleFixedHeightChange(e.target.value)} />
              </label>
            </div>
          </div>
        </div>
        <div className="btn-group btn-group-sm">
          <button className={`h-draw btn btn-default ${drawingType === 'circle' ? 'active' : ''}`} type="button" data-type="circle" title="Draw a new circle (keyboard shortcut: c)" onClick={() => handleDrawingTypeChange('circle')}>
            <span className="icon-circle-empty"></span>Circle
          </button>
        </div>
        <div className="btn-group btn-group-sm">
          <button className={`h-draw btn btn-default ${drawingType === 'polygon' ? 'active' : ''}`} type="button" data-type="polygon" title="Draw a new polygon (keyboard shortcut: p)" onClick={() => handleDrawingTypeChange('polygon')}>
            <span className="icon-draw-polygon"></span>Polygon
          </button>
        </div>
        <div className="btn-group btn-group-sm">
          <button className={`h-draw btn btn-default ${drawingType === 'line' ? 'active' : ''}`} type="button" data-type="line" title="Draw a new line (keyboard shortcut: l)" onClick={() => handleDrawingTypeChange('line')}>
            <span className="icon-pencil"></span>Line
          </button>
        </div>
        <div className="btn-group btn-group-sm">
          <button className={`h-draw btn btn-default ${drawingType === 'brush' ? 'active' : ''}`} type="button" data-type="brush" title="Draw with a brush (keyboard shortcut: b)" onClick={() => handleDrawingTypeChange('brush')} shape={opts.brush_shape}>
            <span className="shape square">
              <span className="icon-check-empty"></span>
            </span>
            <span className="shape circle">
              <span className="icon-circle-empty"></span>
            </span>
            Brush
          </button>
          <button className="btn btn-default h-dropdown-title h-brush-dropdown" type="button" data-target="#h-brush-controls">
            <i className="icon-down-open"></i>
          </button>
          <div className="h-brush-controls collapse">
            <div className="form-group input-sm">
              <label className="radio-inline">
                <input className="h-brush-shape h-brush-square" type="radio" name="h-brush-shape" checked={opts.brush_shape !== 'circle'} shape="square" next_shape="circle" onChange={() => handleBrushShapeChange('square')} />
                Square
              </label>
              <label className="radio-inline">
                <input className="h-brush-shape h-brush-circle" type="radio" name="h-brush-shape" checked={opts.brush_shape === 'circle'} shape="circle" next_shape="square" onChange={() => handleBrushShapeChange('circle')} />
                Circle
              </label>
            </div>
            <div className="form-group input-sm">
              <label>
                Size
                <input className="h-brush-size" type="number" min="1" value={opts.brush_size} onChange={(e) => handleBrushSizeChange(e.target.value)} />
              </label>
            </div>
            <div className="form-group input-sm">
              <label title="If checked, the size is in screen pixels.  If unchecked, the size is in base image pixels">
                <input className="h-brush-screen" type="checkbox" checked={opts.brush_screen} onChange={(e) => handleBrushScreenChange(e.target.checked)} />
                Screen
              </label>
            </div>
          </div>
        </div>
      </div>
      <div className="h-group-count">
        <b className="h-group-count-label">Count:</b>
        <span className="h-group-count-options"></span>
      </div>
      <div className="h-elements-container">
        <AnnotationCanvas />
        {/* Include other elements here */}
      </div>
    </div>
  );
};

export default DrawWidget;