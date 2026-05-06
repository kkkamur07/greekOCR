// /Users/krishuagarwal/Desktop/Programming/python/greek-ocr/frontend/src/components/ImageCanvas/hooks/useDrawing.ts

import { useState, useCallback, useEffect, RefObject } from 'react';
import { message } from 'antd';
import { DrawMode, Region, Point } from '../../../types';

export const useDrawing = (
  drawMode: DrawMode,
  regions: Region[],
  onRegionDrawn: ((region: Region) => void) | undefined,
  imageRef: RefObject<HTMLImageElement>
) => {
  // Box drawing states
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPoint, setStartPoint] = useState<Point | null>(null);
  const [currentRect, setCurrentRect] = useState<{ x: number; y: number; width: number; height: number } | null>(null);
  
  // Polygon drawing states
  const [drawingPoints, setDrawingPoints] = useState<Point[]>([]);
  const [currentMousePos, setCurrentMousePos] = useState<Point | null>(null);

  // Reset when mode changes
  useEffect(() => {
    setIsDrawing(false);
    setDrawingPoints([]);
    setCurrentMousePos(null);
    setStartPoint(null);
    setCurrentRect(null);
  }, [drawMode]);

  const getRelativeCoordinates = useCallback((e: React.MouseEvent): Point | null => {
    if (!imageRef.current) return null;
    
    const rect = imageRef.current.getBoundingClientRect();
    const scaleX = imageRef.current.naturalWidth / rect.width;
    const scaleY = imageRef.current.naturalHeight / rect.height;
    
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    
    return { x, y };
  }, [imageRef]);

  // ============ BOX DRAWING ============
  const handleBoxDrawStart = useCallback((coords: Point) => {
    setIsDrawing(true);
    setStartPoint(coords);
    setCurrentRect({ x: coords.x, y: coords.y, width: 0, height: 0 });
  }, []);

  const handleBoxDrawMove = useCallback((coords: Point) => {
    if (!startPoint) return;
    
    const width = coords.x - startPoint.x;
    const height = coords.y - startPoint.y;
    
    setCurrentRect({
      x: width > 0 ? startPoint.x : coords.x,
      y: height > 0 ? startPoint.y : coords.y,
      width: Math.abs(width),
      height: Math.abs(height),
    });
  }, [startPoint]);

  const handleBoxDrawEnd = useCallback(() => {
    if (!currentRect || !onRegionDrawn) {
      setIsDrawing(false);
      setStartPoint(null);
      setCurrentRect(null);
      return;
    }

    if (currentRect.width > 10 && currentRect.height > 10) {
      const newRegion: Region = {
        id: regions.length > 0 ? Math.max(...regions.map(r => r.id)) + 1 : 0,
        boundary: [
          [currentRect.x, currentRect.y],
          [currentRect.x + currentRect.width, currentRect.y],
          [currentRect.x + currentRect.width, currentRect.y + currentRect.height],
          [currentRect.x, currentRect.y + currentRect.height],
        ],
        bbox: [
          currentRect.x,
          currentRect.y,
          currentRect.x + currentRect.width,
          currentRect.y + currentRect.height,
        ],
      };
      
      onRegionDrawn(newRegion);
      message.success('Box region added');
    }
    
    setIsDrawing(false);
    setStartPoint(null);
    setCurrentRect(null);
  }, [currentRect, onRegionDrawn, regions]);

  // ============ POLYGON DRAWING ============
  const handlePolygonClick = useCallback((coords: Point) => {
    if (!isDrawing) {
      setIsDrawing(true);
      setDrawingPoints([coords]);
      message.info('Click to add points. Double-click or press Enter to finish.');
    } else {
      setDrawingPoints([...drawingPoints, coords]);
    }
  }, [isDrawing, drawingPoints]);

  const handlePolygonComplete = useCallback(() => {
    if (drawingPoints.length >= 3 && onRegionDrawn) {
      const boundary = drawingPoints.map(p => [p.x, p.y]);
      
      const xs = drawingPoints.map(p => p.x);
      const ys = drawingPoints.map(p => p.y);
      const bbox: [number, number, number, number] = [
        Math.min(...xs),
        Math.min(...ys),
        Math.max(...xs),
        Math.max(...ys),
      ];
      
      const newRegion: Region = {
        id: regions.length > 0 ? Math.max(...regions.map(r => r.id)) + 1 : 0,
        boundary,
        bbox,
      };
      
      onRegionDrawn(newRegion);
      message.success('Polygon region added');
      
      setIsDrawing(false);
      setDrawingPoints([]);
      setCurrentMousePos(null);
    } else if (drawingPoints.length < 3) {
      message.warning('Need at least 3 points to create a polygon');
    }
  }, [drawingPoints, onRegionDrawn, regions]);

  const handleCancelPolygon = useCallback(() => {
    setIsDrawing(false);
    setDrawingPoints([]);
    setCurrentMousePos(null);
    message.info('Polygon drawing cancelled');
  }, []);

  // Keyboard shortcuts for polygon
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (drawMode === 'polygon' && isDrawing) {
        if (e.key === 'Escape') {
          handleCancelPolygon();
        } else if (e.key === 'Enter' && drawingPoints.length >= 3) {
          handlePolygonComplete();
        }
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [drawMode, isDrawing, drawingPoints, handleCancelPolygon, handlePolygonComplete]);

  // ============ MAIN HANDLERS ============
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const coords = getRelativeCoordinates(e);
    if (!coords) return;

    if (drawMode === 'box') {
      handleBoxDrawStart(coords);
    }
  }, [drawMode, getRelativeCoordinates, handleBoxDrawStart]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const coords = getRelativeCoordinates(e);
    if (!coords) return;
    
    if (drawMode === 'polygon' && isDrawing) {
      setCurrentMousePos(coords);
    }
    
    if (drawMode === 'box' && isDrawing && startPoint) {
      handleBoxDrawMove(coords);
    }
  }, [drawMode, isDrawing, startPoint, getRelativeCoordinates, handleBoxDrawMove]);

  const handleMouseUp = useCallback(() => {
    if (drawMode === 'box' && isDrawing) {
      handleBoxDrawEnd();
    }
  }, [drawMode, isDrawing, handleBoxDrawEnd]);

  const handleClick = useCallback((e: React.MouseEvent) => {
    const coords = getRelativeCoordinates(e);
    if (!coords) return;
    
    if (drawMode === 'polygon') {
      handlePolygonClick(coords);
    }
  }, [drawMode, getRelativeCoordinates, handlePolygonClick]);

  const handleDoubleClick = useCallback((e: React.MouseEvent) => {
    if (drawMode === 'polygon' && isDrawing) {
      e.preventDefault();
      handlePolygonComplete();
    }
  }, [drawMode, isDrawing, handlePolygonComplete]);

  return {
    isDrawing,
    drawingPoints,
    currentMousePos,
    currentRect,
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
    handleClick,
    handleDoubleClick,
  };
};