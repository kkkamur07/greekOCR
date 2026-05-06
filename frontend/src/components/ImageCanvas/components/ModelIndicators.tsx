import React from 'react';
import { DrawMode } from '../../../types';

interface ModeIndicatorsProps {
  drawMode: DrawMode;
  editMode: 'none' | 'vertices';
  isDrawing: boolean;
  pointCount: number;
}

export const ModeIndicators: React.FC<ModeIndicatorsProps> = ({
  drawMode,
  editMode,
  isDrawing,
  pointCount,
}) => {
  return (
    <>
      {drawMode !== 'none' && (
        <div
          style={{
            position: 'absolute',
            top: '60px',
            left: '10px',
            background: 'rgba(24, 144, 255, 0.9)',
            color: 'white',
            padding: '8px 16px',
            borderRadius: '4px',
            zIndex: 1000,
            fontWeight: 'bold',
          }}
        >
          {drawMode === 'box' && '📦 Box Drawing Mode - Click & Drag'}
          {drawMode === 'polygon' && isDrawing && 
            `🔷 Polygon Mode - ${pointCount} points (ESC to cancel, Enter to finish)`}
          {drawMode === 'polygon' && !isDrawing && '🔷 Polygon Mode - Click to start'}
        </div>
      )}
      
      {editMode === 'vertices' && (
        <div
          style={{
            position: 'absolute',
            top: '100px',
            left: '10px',
            background: 'rgba(250, 140, 22, 0.9)',
            color: 'white',
            padding: '8px 16px',
            borderRadius: '4px',
            zIndex: 1000,
            fontWeight: 'bold',
          }}
        >
          ✏️ Vertex Edit Mode - Drag vertices to adjust
        </div>
      )}
    </>
  );
};