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
            top: '10px',
            left: '50%',
            transform: 'translateX(-50%)',
            background: 'rgba(24, 144, 255, 0.95)',
            color: 'white',
            padding: '12px 24px',
            borderRadius: '8px',
            zIndex: 1000,
            fontWeight: 'bold',
            boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
            fontSize: '14px',
          }}
        >
          {drawMode === 'box' && '📦 Box Drawing Mode - Click & Drag'}
          {drawMode === 'polygon' && isDrawing && 
            `🔷 Polygon Mode - ${pointCount} points (Double-click, Enter to finish, ESC to cancel)`}
          {drawMode === 'polygon' && !isDrawing && '🔷 Polygon Mode - Click to start'}
        </div>
      )}
      
      {editMode === 'vertices' && (
        <div
          style={{
            position: 'absolute',
            top: drawMode !== 'none' ? '60px' : '10px',
            left: '50%',
            transform: 'translateX(-50%)',
            background: 'rgba(250, 140, 22, 0.95)',
            color: 'white',
            padding: '12px 24px',
            borderRadius: '8px',
            zIndex: 1000,
            fontWeight: 'bold',
            boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
            fontSize: '14px',
          }}
        >
          ✏️ Vertex Edit Mode - Drag vertices to adjust region
        </div>
      )}
    </>
  );
};
