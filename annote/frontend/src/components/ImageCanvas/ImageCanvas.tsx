import { useState, useCallback } from 'react';
import type { Region, DrawMode, EditingSettings, PointTuple } from '../../types';
import { Button, Space, Empty, message, Tooltip, Tag } from 'antd';
import {
  PlusOutlined, DeleteOutlined, EditOutlined, BorderOutlined,
  StopOutlined, UndoOutlined, RedoOutlined, SettingOutlined
} from '@ant-design/icons';
import { useKeyboardShortcuts, KEYBOARD_SHORTCUTS } from '../../hooks/useKeyboardShortcuts';
import ImageViewer from './components/ImageViewer';
import { SettingsPanel } from './components/SettingsPanel';
import './ImageCanvas.css';

interface ImageCanvasProps {
  imageUrl: string | null;
  imageDimensions: { width: number; height: number };
  regions: Region[];
  selectedRegionId: number | null;
  onSelectRegion: (id: number | null) => void;
  onAddRegion: (region: Region) => void;
  onUpdateRegion: (region: Region) => void;
  onDeleteRegion: (id: number) => void;
  onTranscribeRegion: (regionId: number) => void;
  onUndo?: () => void;
  onRedo?: () => void;
  canUndo?: boolean;
  canRedo?: boolean;
  readOnly?: boolean;
}

const ImageCanvas: React.FC<ImageCanvasProps> = ({
  imageUrl,
  regions,
  selectedRegionId,
  onSelectRegion,
  onAddRegion,
  onUpdateRegion,
  onDeleteRegion,
  onTranscribeRegion,
  onUndo,
  onRedo,
  canUndo = false,
  canRedo = false,
  readOnly = false,
}) => {
  const [drawMode, setDrawMode] = useState<DrawMode>('none');
  const [editMode, setEditMode] = useState<'none' | 'vertices'>('none');
  const [settingsVisible, setSettingsVisible] = useState(false);
  const [settings, setSettings] = useState<EditingSettings>({
    showBoundingBoxes: true,
    vertexSize: 1.5,
    moveStep: 5,
  });

  const handleRegionDrawn = useCallback((region: Region) => {
    onAddRegion(region);
    setDrawMode('none');
  }, [onAddRegion]);

  const handleRegionUpdated = useCallback((region: Region) => {
    onUpdateRegion(region);
  }, [onUpdateRegion]);

  const handleDeleteSelected = useCallback(() => {
    if (selectedRegionId !== null) {
      onDeleteRegion(selectedRegionId);
      message.success('Region deleted');
    }
  }, [selectedRegionId, onDeleteRegion]);

  const toggleDrawMode = useCallback((mode: DrawMode) => {
    if (drawMode === mode) {
      setDrawMode('none');
    } else {
      setDrawMode(mode);
      setEditMode('none');
    }
  }, [drawMode]);

  const toggleEditMode = useCallback(() => {
    if (editMode === 'vertices') {
      setEditMode('none');
    } else {
      setEditMode('vertices');
      setDrawMode('none');
    }
  }, [editMode]);

  const cancelMode = useCallback(() => {
    setDrawMode('none');
    setEditMode('none');
  }, []);

  // Arrow key movement handlers
  const moveRegion = useCallback((dx: number, dy: number) => {
    if (selectedRegionId === null) {
      message.warning('Please select a region first');
      return;
    }

    const region = regions.find(r => r.id === selectedRegionId);
    if (!region) return;

    const newBoundary = region.boundary.map(
      ([x, y]): PointTuple => [x + dx, y + dy],
    );
    const newBbox: [number, number, number, number] = [
      region.bbox[0] + dx,
      region.bbox[1] + dy,
      region.bbox[2] + dx,
      region.bbox[3] + dy,
    ];

    const updatedRegion: Region = {
      ...region,
      boundary: newBoundary,
      bbox: newBbox,
    };

    onUpdateRegion(updatedRegion);
  }, [selectedRegionId, regions, onUpdateRegion]);

  // Keyboard shortcuts
  useKeyboardShortcuts({
    onDrawBox: readOnly ? undefined : () => imageUrl && toggleDrawMode('box'),
    onDrawPolygon: readOnly ? undefined : () => imageUrl && toggleDrawMode('polygon'),
    onEditVertices: readOnly ? undefined : () => imageUrl && regions.length > 0 && toggleEditMode(),
    onDelete: readOnly ? undefined : handleDeleteSelected,
    onEscape: readOnly ? undefined : cancelMode,
    onUndo: readOnly ? undefined : onUndo,
    onRedo: readOnly ? undefined : onRedo,
    onMoveUp: readOnly ? undefined : () => moveRegion(0, -settings.moveStep),
    onMoveDown: readOnly ? undefined : () => moveRegion(0, settings.moveStep),
    onMoveLeft: readOnly ? undefined : () => moveRegion(-settings.moveStep, 0),
    onMoveRight: readOnly ? undefined : () => moveRegion(settings.moveStep, 0),
  });

  return (
    <div className="image-canvas-container">
      {!readOnly && (
      <div className="canvas-toolbar">
        <Space wrap size="middle">
          {/* Drawing Tools */}
          <Space.Compact>
            <Tooltip title={`Draw Box (${KEYBOARD_SHORTCUTS.DRAW_BOX})`}>
              <Button
                icon={<BorderOutlined />}
                onClick={() => toggleDrawMode('box')}
                type={drawMode === 'box' ? 'primary' : 'default'}
                disabled={!imageUrl}
              >
                <Tag>{KEYBOARD_SHORTCUTS.DRAW_BOX}</Tag>
              </Button>
            </Tooltip>
            <Tooltip title={`Draw Polygon (${KEYBOARD_SHORTCUTS.DRAW_POLYGON})`}>
              <Button
                icon={<PlusOutlined />}
                onClick={() => toggleDrawMode('polygon')}
                type={drawMode === 'polygon' ? 'primary' : 'default'}
                disabled={!imageUrl}
              >
                Polygon <Tag>{KEYBOARD_SHORTCUTS.DRAW_POLYGON}</Tag>
              </Button>
            </Tooltip>
          </Space.Compact>

          {/* Edit Tool */}
          <Tooltip title={`Edit Vertices (${KEYBOARD_SHORTCUTS.EDIT_VERTICES})`}>
            <Button
              icon={<EditOutlined />}
              onClick={toggleEditMode}
              type={editMode === 'vertices' ? 'primary' : 'default'}
              disabled={!imageUrl || regions.length === 0}
            >
              Edit Vertices <Tag>{KEYBOARD_SHORTCUTS.EDIT_VERTICES}</Tag>
            </Button>
          </Tooltip>

          {/* Cancel Button */}
          {(drawMode !== 'none' || editMode !== 'none') && (
            <Tooltip title={KEYBOARD_SHORTCUTS.CANCEL}>
              <Button
                icon={<StopOutlined />}
                onClick={cancelMode}
                danger
              >
                Cancel <Tag color="red">{KEYBOARD_SHORTCUTS.CANCEL}</Tag>
              </Button>
            </Tooltip>
          )}

          {/* Undo/Redo */}
          <Space.Compact>
            <Tooltip title={KEYBOARD_SHORTCUTS.UNDO}>
              <Button
                icon={<UndoOutlined />}
                onClick={onUndo}
                disabled={!canUndo}
              >
                Undo
              </Button>
            </Tooltip>
            <Tooltip title={KEYBOARD_SHORTCUTS.REDO}>
              <Button
                icon={<RedoOutlined />}
                onClick={onRedo}
                disabled={!canRedo}
              >
                Redo
              </Button>
            </Tooltip>
          </Space.Compact>

          {/* Delete */}
          <Tooltip title={KEYBOARD_SHORTCUTS.DELETE}>
            <Button
              icon={<DeleteOutlined />}
              danger
              onClick={handleDeleteSelected}
              disabled={selectedRegionId === null}
            >
              Delete
            </Button>
          </Tooltip>

          {/* Settings Button */}
          <Tooltip title="Editing Settings">
            <Button
              icon={<SettingOutlined />}
              onClick={() => setSettingsVisible(true)}
            >
              Settings
            </Button>
          </Tooltip>

          {/* Status */}
          <span style={{ marginLeft: 8, color: '#666', fontSize: '13px' }}>
            Regions: <strong>{regions.length}</strong> |
            Selected: <strong>{selectedRegionId !== null ? `#${selectedRegionId}` : 'None'}</strong>
          </span>
        </Space>
      </div>
      )}

      <div className="canvas-wrapper">
        {!imageUrl ? (
          <Empty description="No image loaded" />
        ) : (
          <ImageViewer
            imageUrl={imageUrl}
            regions={regions}
            selectedRegionId={selectedRegionId}
            onSelectRegion={onSelectRegion}
            onRegionDrawn={handleRegionDrawn}
            onRegionUpdated={handleRegionUpdated}
            onTranscribeRegion={onTranscribeRegion}
            drawMode={drawMode}
            editMode={editMode}
            settings={settings}
          />
        )}
      </div>

      {!readOnly && (
        <SettingsPanel
          visible={settingsVisible}
          onClose={() => setSettingsVisible(false)}
          settings={settings}
          onSettingsChange={setSettings}
        />
      )}
    </div>
  );
};

export default ImageCanvas;