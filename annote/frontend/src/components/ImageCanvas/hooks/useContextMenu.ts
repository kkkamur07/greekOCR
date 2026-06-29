// /Users/krishuagarwal/Desktop/Programming/python/greek-ocr/frontend/src/components/ImageCanvas/hooks/useContextMenu.ts

import { useState, useCallback } from 'react';
import { message } from 'antd';

export const useContextMenu = (
  onTranscribeRegion?: (regionId: number) => void
) => {
  const [contextMenu, setContextMenu] = useState<{
    visible: boolean;
    x: number;
    y: number;
    regionId: number | null;
  }>({ visible: false, x: 0, y: 0, regionId: null });

  const open = useCallback((x: number, y: number, regionId: number) => {
    setContextMenu({ visible: true, x, y, regionId });
  }, []);

  const close = useCallback(() => {
    setContextMenu(prev => ({ ...prev, visible: false }));
  }, []);

  const onTranscribe = useCallback(() => {
    if (contextMenu.regionId !== null && onTranscribeRegion) {
      onTranscribeRegion(contextMenu.regionId);
    }
    close();
  }, [contextMenu.regionId, onTranscribeRegion, close]);

  const onEditVertices = useCallback(() => {
    close();
    message.info('Vertex editing mode - drag vertices to adjust region');
  }, [close]);

  return {
    visible: contextMenu.visible,
    x: contextMenu.x,
    y: contextMenu.y,
    regionId: contextMenu.regionId,
    open,
    close,
    onTranscribe,
    onEditVertices,
  };
};