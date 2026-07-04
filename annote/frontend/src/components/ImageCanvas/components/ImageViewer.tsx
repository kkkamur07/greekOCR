import { useRef, useState } from 'react';
import { TransformWrapper, TransformComponent, type ReactZoomPanPinchRef } from 'react-zoom-pan-pinch';
import type { Region, DrawMode, EditingSettings } from '../../../types';
import { ZoomControls } from './ZoomControls';
import { ModeIndicators } from './ModeIndicators';
import { RegionsOverlay } from './RegionOveraly';
import { ContextMenu } from './ContextMenu';
import { useDrawing } from '../hooks/useDrawing';
import { useVertexEditing } from '../hooks/useVertexEditing';
import { useContextMenu } from '../hooks/useContextMenu';

import './ImageViewer.css';

interface ImageViewerProps {
  imageUrl: string;
  regions: Region[];
  selectedRegionId: number | null;
  onSelectRegion: (id: number | null) => void;
  onRegionDrawn?: (region: Region) => void;
  onRegionUpdated?: (region: Region) => void;
  onTranscribeRegion?: (regionId: number) => void;
  drawMode: DrawMode;
  editMode: 'none' | 'vertices';
  settings: EditingSettings;
}

const ImageViewer: React.FC<ImageViewerProps> = ({
  imageUrl,
  regions,
  selectedRegionId,
  onSelectRegion,
  onRegionDrawn,
  onRegionUpdated,
  onTranscribeRegion,
  drawMode,
  editMode,
  settings,
}) => {
  const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number } | null>(null);
  const [zoomLevel, setZoomLevel] = useState(1);

  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const transformRef = useRef<ReactZoomPanPinchRef>(null!);

  // Custom hooks for different features
  const drawing = useDrawing(drawMode, regions, onRegionDrawn, imageRef);
  const vertexEditing = useVertexEditing(editMode, regions, onRegionUpdated, onSelectRegion, imageRef);
  const contextMenu = useContextMenu(onTranscribeRegion);

  const handleImageLoad = () => {
    if (imageRef.current) {
      setImageDimensions({
        width: imageRef.current.naturalWidth,
        height: imageRef.current.naturalHeight,
      });
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (contextMenu.visible) {
      contextMenu.close();
      return;
    }

    drawing.handleMouseDown(e);
    vertexEditing.handleMouseDown();
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    drawing.handleMouseMove(e);
    vertexEditing.handleMouseMove(e);
  };

  const handleMouseUp = () => {
    drawing.handleMouseUp();
    vertexEditing.handleMouseUp();
  };

  const handleClick = (e: React.MouseEvent) => {
    drawing.handleClick(e);
  };

  const handleDoubleClick = (e: React.MouseEvent) => {
    drawing.handleDoubleClick(e);
  };

  const handleRegionClick = (regionId: number, e: React.MouseEvent) => {
    e.stopPropagation();
    if (drawMode !== 'none') return;
    onSelectRegion(regionId);
  };

  const handleRegionRightClick = (regionId: number, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onSelectRegion(regionId);
    contextMenu.open(e.clientX, e.clientY, regionId);
  };

  const isPanningEnabled = drawMode === 'none' &&
                          editMode === 'none' &&
                          !drawing.isDrawing &&
                          !vertexEditing.isDragging;

  return (
    <div className="image-viewer-container" ref={containerRef}>
      <ContextMenu {...contextMenu} />

      <TransformWrapper
        ref={transformRef}
        initialScale={1}
        minScale={0.3}
        maxScale={10}
        centerOnInit={true}
        wheel={{ step: 0.15, smoothStep: 0.01 }}
        doubleClick={{ disabled: drawMode !== 'none', step: 0.7 }}
        panning={{ disabled: !isPanningEnabled, velocityDisabled: false }}
        pinch={{ disabled: drawMode !== 'none', step: 5 }}
        velocityAnimation={{ disabled: false, sensitivity: 1, animationTime: 400 }}
        onTransformed={(ref) => setZoomLevel(ref.state.scale)}
      >
        {() => (
          <>
            <ZoomControls
              zoomLevel={zoomLevel}
              transformRef={transformRef}
            />

            <ModeIndicators
              drawMode={drawMode}
              editMode={editMode}
              isDrawing={drawing.isDrawing}
              pointCount={drawing.drawingPoints.length}
            />

            <TransformComponent
              wrapperStyle={{ width: '100%', height: '100%', touchAction: 'none' }}
              contentStyle={{ width: '100%', height: '100%' }}
            >
              <div
                className="image-wrapper"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onClick={handleClick}
                onDoubleClick={handleDoubleClick}
                style={{
                  cursor: drawMode === 'box' || drawMode === 'polygon' ? 'crosshair' :
                         editMode === 'vertices' ? 'move' :
                         isPanningEnabled ? 'grab' : 'default',
                  touchAction: 'none',
                }}
              >
                <img
                  ref={imageRef}
                  src={imageUrl}
                  alt="OCR Document"
                  draggable={false}
                  onLoad={handleImageLoad}
                  style={{
                    display: 'block',
                    width: '100%',
                    height: 'auto',
                    touchAction: 'none',
                    userSelect: 'none',
                  }}
                />

                {imageDimensions && (
                  <RegionsOverlay
                    imageDimensions={imageDimensions}
                    regions={regions}
                    selectedRegionId={selectedRegionId}
                    zoomLevel={zoomLevel}
                    drawMode={drawMode}
                    editMode={editMode}
                    drawingState={drawing}
                    vertexEditingState={vertexEditing}
                    settings={settings}
                    onRegionClick={handleRegionClick}
                    onRegionRightClick={handleRegionRightClick}
                  />
                )}
              </div>
            </TransformComponent>
          </>
        )}
      </TransformWrapper>
    </div>
  );
};

export default ImageViewer;