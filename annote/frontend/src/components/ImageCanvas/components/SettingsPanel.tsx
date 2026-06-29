import React from 'react';
import { Drawer, Switch, Slider, Typography, Divider, Space } from 'antd';
import { SettingOutlined } from '@ant-design/icons';
import { EditingSettings } from '../../../types';

const { Text, Title } = Typography;

interface SettingsPanelProps {
  visible: boolean;
  onClose: () => void;
  settings: EditingSettings;
  onSettingsChange: (settings: EditingSettings) => void;
}

export const SettingsPanel: React.FC<SettingsPanelProps> = ({
  visible,
  onClose,
  settings,
  onSettingsChange,
}) => {
  const handleToggleBoundingBoxes = (checked: boolean) => {
    onSettingsChange({ ...settings, showBoundingBoxes: checked });
  };

  const handleVertexSizeChange = (value: number) => {
    onSettingsChange({ ...settings, vertexSize: value });
  };

  const handleMoveStepChange = (value: number) => {
    onSettingsChange({ ...settings, moveStep: value });
  };

  return (
    <Drawer
      title={
        <Space>
          <SettingOutlined />
          <span>Editing Settings</span>
        </Space>
      }
      placement="right"
      onClose={onClose}
      open={visible}
      width={350}
    >
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {/* Display Options */}
        <div>
          <Title level={5}>Display Options</Title>
          <Space direction="vertical" style={{ width: '100%' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Text>Show Bounding Boxes</Text>
              <Switch
                checked={settings.showBoundingBoxes}
                onChange={handleToggleBoundingBoxes}
              />
            </div>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              Toggle the blue dotted rectangles around regions
            </Text>
          </Space>
        </div>

        <Divider />

        {/* Vertex Settings */}
        <div>
          <Title level={5}>Vertex Editing</Title>
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            <div>
              <Text>Vertex Size: <strong>{settings.vertexSize}x</strong></Text>
              <Slider
                min={1}
                max={3}
                step={0.5}
                value={settings.vertexSize}
                onChange={handleVertexSizeChange}
                marks={{
                  1: 'Small',
                  2: 'Medium',
                  3: 'Large',
                }}
              />
              <Text type="secondary" style={{ fontSize: '12px' }}>
                Size of vertex handles in edit mode
              </Text>
            </div>
          </Space>
        </div>

        <Divider />

        {/* Movement Settings */}
        <div>
          <Title level={5}>Region Movement</Title>
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            <div>
              <Text>Arrow Key Step: <strong>{settings.moveStep}px</strong></Text>
              <Slider
                min={1}
                max={20}
                value={settings.moveStep}
                onChange={handleMoveStepChange}
                marks={{
                  1: '1px',
                  5: '5px',
                  10: '10px',
                  20: '20px',
                }}
              />
              <Text type="secondary" style={{ fontSize: '12px' }}>
                Distance moved with arrow keys
              </Text>
            </div>
          </Space>
        </div>

        <Divider />

        {/* Keyboard Shortcuts Reference */}
        <div>
          <Title level={5}>Keyboard Shortcuts</Title>
          <Space direction="vertical" style={{ width: '100%', fontSize: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <Text type="secondary">Draw Box:</Text>
              <Text code>B</Text>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <Text type="secondary">Draw Polygon:</Text>
              <Text code>P</Text>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <Text type="secondary">Edit Vertices:</Text>
              <Text code>V</Text>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <Text type="secondary">Move Region:</Text>
              <Text code>↑ ↓ ← →</Text>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <Text type="secondary">Delete:</Text>
              <Text code>Del</Text>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <Text type="secondary">Undo:</Text>
              <Text code>Ctrl+Z</Text>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <Text type="secondary">Redo:</Text>
              <Text code>Ctrl+Shift+Z</Text>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <Text type="secondary">Cancel:</Text>
              <Text code>Esc</Text>
            </div>
          </Space>
        </div>
      </Space>
    </Drawer>
  );
};
