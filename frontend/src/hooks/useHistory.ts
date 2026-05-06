import { useState, useCallback, useEffect } from 'react';
import { Region, Transcription, HistoryState } from '../types';
import { message } from 'antd';

const MAX_HISTORY = 50;

export const useHistory = (
  initialRegions: Region[] = [],
  initialTranscriptions: Transcription[] = [],
  initialSelectedId: number | null = null
) => {
  const [history, setHistory] = useState<HistoryState[]>([{
    regions: initialRegions,
    transcriptions: initialTranscriptions,
    selectedRegionId: initialSelectedId,
  }]);
  const [currentIndex, setCurrentIndex] = useState(0);

  const currentState = history[currentIndex];

  // Add to history
  const pushState = useCallback((
    regions: Region[],
    transcriptions: Transcription[],
    selectedRegionId: number | null
  ) => {
    setHistory(prev => {
      // Remove any states after current index
      const newHistory = prev.slice(0, currentIndex + 1);
      
      // Add new state
      newHistory.push({ regions, transcriptions, selectedRegionId });
      
      // Limit history size
      if (newHistory.length > MAX_HISTORY) {
        newHistory.shift();
        setCurrentIndex(prev => prev - 1);
      }
      
      return newHistory;
    });
    
    setCurrentIndex(prev => Math.min(prev + 1, MAX_HISTORY - 1));
  }, [currentIndex]);

  // Undo
  const undo = useCallback(() => {
    if (currentIndex > 0) {
      setCurrentIndex(prev => prev - 1);
      message.info('Undo');
      return true;
    }
    message.warning('Nothing to undo');
    return false;
  }, [currentIndex]);

  // Redo
  const redo = useCallback(() => {
    if (currentIndex < history.length - 1) {
      setCurrentIndex(prev => prev + 1);
      message.info('Redo');
      return true;
    }
    message.warning('Nothing to redo');
    return false;
  }, [currentIndex, history.length]);

  const canUndo = currentIndex > 0;
  const canRedo = currentIndex < history.length - 1;

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+Z or Cmd+Z for undo
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        undo();
      }
      // Ctrl+Shift+Z or Cmd+Shift+Z or Ctrl+Y for redo
      else if (((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'z') || 
               (e.ctrlKey && e.key === 'y')) {
        e.preventDefault();
        redo();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [undo, redo]);

  return {
    currentState,
    pushState,
    undo,
    redo,
    canUndo,
    canRedo,
  };
};
