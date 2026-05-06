import React from 'react';
import { Region, DrawMode, EditingSettings } from '../../../types';

interface RegionsOverlayProps {
  imageDimensions: { width: number; height: number };
  regions: Region[];
  selectedRegionId: number | null;
  zoomLevel: number;
  drawMode: DrawMode;
  editMode: 'none' | 'vertices';
  drawingState: any;
  vertexEditingState: any;
  settings: EditingSettings;
  onRegionClick: (regionId: number, e: React.MouseEvent) => void;
  onRegionRightClick: (regionId: number, e: React.MouseEvent) => void;
}

export const RegionsOverlay: React.FC<RegionsOverlayProps> = ({
  imageDimensions,
  regions,
  selectedRegionId,
  zoomLevel,
  drawMode,
  editMode,
  drawingState,
  vertexEditingState,
  settings,
  onRegionClick,
  onRegionRightClick,
}) => {
  const getBoundaryToRender = (region: Region) => {
    if (vertexEditingState.editingRegionId === region.id && vertexEditingState.tempBoundary) {
      return vertexEditingState.tempBoundary;
    }
    return region.boundary;
  };

  // Calculate responsive sizes based on zoom
  const getStrokeWidth = (base: number) => Math.max(base / zoomLevel, 1);
  const getVertexRadius = () => Math.max((8 * settings.vertexSize) / zoomLevel, 4);
  const getFontSize = () => Math.max(18 / zoomLevel, 12);

  return (
    <svg
      className="regions-overlay"
      viewBox={`0 0 ${imageDimensions.width} ${imageDimensions.height}`}
      preserveAspectRatio="none"
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        touchAction: 'none',
      }}
    >
      {/* Existing regions */}
      {regions.map((region) => {
        const isSelected = region.id === selectedRegionId;
        const boundary = getBoundaryToRender(region);
        const points = boundary.map(([x, y]) => `${x},${y}`).join(' ');
        const [x1, y1, x2, y2] = region.bbox;
        
        return (
          <g key={region.id}>
            {/* Polygon fill and stroke - THICKER */}
            <polygon
              points={points}
              fill="rgba(82, 196, 26, 0.15)"
              stroke={isSelected ? '#fadb14' : '#52c41a'}
              strokeWidth={getStrokeWidth(isSelected ? 5 : 3)}
              style={{ 
                pointerEvents: drawMode === 'none' ? 'all' : 'none',
                cursor: 'pointer',
              }}
              onClick={(e) => onRegionClick(region.id, e as any)}
              onContextMenu={(e) => onRegionRightClick(region.id, e as any)}
            />
            
            {/* Bounding box (dashed) - THICKER - only show if enabled */}
            {settings.showBoundingBoxes && (
              <rect
                x={x1}
                y={y1}
                width={x2 - x1}
                height={y2 - y1}
                fill="transparent"
                stroke={isSelected ? '#fadb14' : '#1890ff'}
                strokeWidth={getStrokeWidth(2)}
                strokeDasharray={`${8 / zoomLevel},${4 / zoomLevel}`}
                style={{ pointerEvents: 'none' }}
              />
            )}
            
            {/* Region label with background */}
            <g>
              <rect
                x={x1 + 2}
                y={y1 + 2}
                width={getFontSize() * 2.5}
                height={getFontSize() * 1.3}
                fill={isSelected ? '#fadb14' : '#52c41a'}
                rx={2}
                opacity={0.8}
              />
              <text
                x={x1 + 5}
                y={y1 + getFontSize() + 2}
                fill="white"
                fontSize={getFontSize()}
                fontWeight="bold"
                style={{ 
                  pointerEvents: 'none', 
                  userSelect: 'none',
                }}
              >
                #{region.id}
              </text>
            </g>
            
            {/* Vertex handles - LARGER & MORE VISIBLE */}
            {editMode === 'vertices' && isSelected && boundary.map(([x, y], idx) => (
              <g key={idx}>
                {/* Outer glow for visibility */}
                <circle
                  cx={x}
                  cy={y}
                  r={getVertexRadius() + 2}
                  fill="rgba(24, 144, 255, 0.3)"
                  style={{ pointerEvents: 'none' }}
                />
                {/* Main vertex circle */}
                <circle
                  cx={x}
                  cy={y}
                  r={getVertexRadius()}
                  fill={vertexEditingState.draggedVertexIndex === idx ? '#ff4d4f' : '#1890ff'}
                  stroke="white"
                  strokeWidth={getStrokeWidth(3)}
                  style={{ 
                    pointerEvents: 'all',
                    cursor: 'move',
                  }}
                  onMouseDown={(e) => vertexEditingState.handleVertexDragStart(region.id, idx, e as any)}
                />
                {/* Vertex number label */}
                <text
                  x={x}
                  y={y - getVertexRadius() - 5}
                  fill="#1890ff"
                  fontSize={getFontSize() * 0.7}
                  fontWeight="bold"
                  textAnchor="middle"
                  style={{ 
                    pointerEvents: 'none',
                    paintOrder: 'stroke',
                    stroke: 'white',
                    strokeWidth: getStrokeWidth(2),
                  }}
                >
                  {idx}
                </text>
              </g>
            ))}
          </g>
        );
      })}
      
      {/* Current box being drawn - THICKER */}
      {drawMode === 'box' && drawingState.currentRect && drawingState.isDrawing && (
        <rect
          x={drawingState.currentRect.x}
          y={drawingState.currentRect.y}
          width={drawingState.currentRect.width}
          height={drawingState.currentRect.height}
          fill="rgba(24, 144, 255, 0.15)"
          stroke="#1890ff"
          strokeWidth={getStrokeWidth(4)}
          strokeDasharray={`${12 / zoomLevel},${6 / zoomLevel}`}
          style={{ pointerEvents: 'none' }}
        />
      )}
      
      {/* Current polygon being drawn - THICKER */}
      {drawMode === 'polygon' && drawingState.isDrawing && drawingState.drawingPoints.length > 0 && (
        <>
          {/* Lines between points - THICKER */}
          {drawingState.drawingPoints.map((point: any, idx: number) => {
            if (idx === 0) return null;
            const prevPoint = drawingState.drawingPoints[idx - 1];
            return (
              <line
                key={`line-${idx}`}
                x1={prevPoint.x}
                y1={prevPoint.y}
                x2={point.x}
                y2={point.y}
                stroke="#1890ff"
                strokeWidth={getStrokeWidth(4)}
              />
            );
          })}
          
          {/* Preview line to current mouse position - THICKER */}
          {drawingState.currentMousePos && (
            <>
              <line
                x1={drawingState.drawingPoints[drawingState.drawingPoints.length - 1].x}
                y1={drawingState.drawingPoints[drawingState.drawingPoints.length - 1].y}
                x2={drawingState.currentMousePos.x}
                y2={drawingState.currentMousePos.y}
                stroke="#1890ff"
                strokeWidth={getStrokeWidth(3)}
                strokeDasharray={`${8 / zoomLevel},${4 / zoomLevel}`}
              />
              
              {/* Closing line preview - THICKER */}
              {drawingState.drawingPoints.length >= 2 && (
                <line
                  x1={drawingState.currentMousePos.x}
                  y1={drawingState.currentMousePos.y}
                  x2={drawingState.drawingPoints[0].x}
                  y2={drawingState.drawingPoints[0].y}
                  stroke="#52c41a"
                  strokeWidth={getStrokeWidth(2)}
                  strokeDasharray={`${8 / zoomLevel},${4 / zoomLevel}`}
                  opacity={0.7}
                />
              )}
            </>
          )}
          
          {/* Points - LARGER */}
          {drawingState.drawingPoints.map((point: any, idx: number) => (
            <g key={`point-${idx}`}>
              {/* Point glow */}
              <circle
                cx={point.x}
                cy={point.y}
                r={getVertexRadius() + 2}
                fill={idx === 0 ? 'rgba(82, 196, 26, 0.3)' : 'rgba(24, 144, 255, 0.3)'}
              />
              {/* Main point */}
              <circle
                cx={point.x}
                cy={point.y}
                r={getVertexRadius()}
                fill={idx === 0 ? '#52c41a' : '#1890ff'}
                stroke="white"
                strokeWidth={getStrokeWidth(3)}
              />
              {/* Point number */}
              <text
                x={point.x}
                y={point.y - getVertexRadius() - 5}
                fill={idx === 0 ? '#52c41a' : '#1890ff'}
                fontSize={getFontSize() * 0.8}
                fontWeight="bold"
                textAnchor="middle"
                style={{ 
                  paintOrder: 'stroke',
                  stroke: 'white',
                  strokeWidth: getStrokeWidth(2),
                }}
              >
                {idx}
              </text>
            </g>
          ))}
        </>
      )}
    </svg>
  );
};