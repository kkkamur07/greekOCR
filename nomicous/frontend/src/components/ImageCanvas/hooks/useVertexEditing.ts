// /Users/krishuagarwal/Desktop/Programming/python/greek-ocr/frontend/src/components/ImageCanvas/hooks/useVertexEditing.ts

import { useState, useCallback, useEffect, type RefObject } from 'react';
import { message } from 'antd';
import { Region, Point, type PointTuple } from '../../../types';

export const useVertexEditing = (
  editMode: 'none' | 'vertices',
  regions: Region[],
  onRegionUpdated: ((region: Region) => void) | undefined,
  onSelectRegion: (id: number | null) => void,
  imageRef: RefObject<HTMLImageElement | null>
) => {
  const [editingRegionId, setEditingRegionId] = useState<number | null>(null);
  const [draggedVertexIndex, setDraggedVertexIndex] = useState<number | null>(null);
  const [tempBoundary, setTempBoundary] = useState<PointTuple[] | null>(null);

  // Reset local drag state when vertex editing is turned off.
  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    if (editMode === 'none') {
      setEditingRegionId(null);
      setDraggedVertexIndex(null);
      setTempBoundary(null);
    }
  }, [editMode]);
  /* eslint-enable react-hooks/set-state-in-effect */

  const getRelativeCoordinates = useCallback((e: React.MouseEvent): Point | null => {
    if (!imageRef.current) return null;

    const rect = imageRef.current.getBoundingClientRect();
    const scaleX = imageRef.current.naturalWidth / rect.width;
    const scaleY = imageRef.current.naturalHeight / rect.height;

    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    return { x, y };
  }, [imageRef]);

  const handleVertexDragStart = useCallback((regionId: number, vertexIndex: number, e: React.MouseEvent) => {
    e.stopPropagation();

    if (editMode !== 'vertices') return;

    const region = regions.find(r => r.id === regionId);
    if (!region) return;

    setEditingRegionId(regionId);
    setDraggedVertexIndex(vertexIndex);
    setTempBoundary([...region.boundary]);
    onSelectRegion(regionId);
  }, [editMode, regions, onSelectRegion]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (draggedVertexIndex === null || !tempBoundary) return;

    const coords = getRelativeCoordinates(e);
    if (!coords) return;

    const newBoundary = [...tempBoundary];
    const nextPoint: PointTuple = [coords.x, coords.y];
    newBoundary[draggedVertexIndex] = nextPoint;
    setTempBoundary(newBoundary);
  }, [draggedVertexIndex, tempBoundary, getRelativeCoordinates]);

  const handleMouseUp = useCallback(() => {
    if (editingRegionId === null || !tempBoundary || !onRegionUpdated) {
      setEditingRegionId(null);
      setDraggedVertexIndex(null);
      setTempBoundary(null);
      return;
    }

    // Calculate new bounding box
    const xs = tempBoundary.map(p => p[0]);
    const ys = tempBoundary.map(p => p[1]);
    const bbox: [number, number, number, number] = [
      Math.min(...xs),
      Math.min(...ys),
      Math.max(...xs),
      Math.max(...ys),
    ];

    const updatedRegion: Region = {
      id: editingRegionId,
      boundary: tempBoundary,
      bbox,
    };

    onRegionUpdated(updatedRegion);
    message.success('Region vertices updated');

    setEditingRegionId(null);
    setDraggedVertexIndex(null);
    setTempBoundary(null);
  }, [editingRegionId, tempBoundary, onRegionUpdated]);

  const handleMouseDown = useCallback(() => {
    // This is handled by handleVertexDragStart
  }, []);

  return {
    editingRegionId,
    draggedVertexIndex,
    tempBoundary,
    isDragging: draggedVertexIndex !== null,
    handleVertexDragStart,
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
  };
};