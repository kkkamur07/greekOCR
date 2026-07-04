import React from 'react';
import type { ReactZoomPanPinchRef } from 'react-zoom-pan-pinch';

interface ZoomControlsProps {
  zoomLevel: number;
  transformRef: React.RefObject<ReactZoomPanPinchRef | null>;
}

export const ZoomControls: React.FC<ZoomControlsProps> = ({ zoomLevel, transformRef }) => {
  const handleZoomIn = () => transformRef.current?.zoomIn();
  const handleZoomOut = () => transformRef.current?.zoomOut();
  const handleReset = () => transformRef.current?.resetTransform();
  const handleFitToScreen = () => transformRef.current?.centerView();

  return (
    <div className="zoom-controls">
      <div className="zoom-level">{Math.round(zoomLevel * 100)}%</div>
      <button onClick={handleZoomIn} title="Zoom In">+</button>
      <button onClick={handleZoomOut} title="Zoom Out">−</button>
      <button onClick={handleFitToScreen} title="Fit to Screen">⊡</button>
      <button onClick={handleReset} title="Reset View">⟲</button>
    </div>
  );
};