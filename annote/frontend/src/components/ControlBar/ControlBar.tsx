import { Upload, Button, Slider, Space, Collapse } from 'antd';
import { UploadOutlined, ScissorOutlined, FileTextOutlined, FilterOutlined } from '@ant-design/icons';
import { useState } from 'react';
import type { UploadFile } from 'antd';
import './ControlBar.css';

interface ControlBarProps {
  onImageUpload: (file: File) => void;
  onBinarize: () => void;
  onSegment: (minArea: number, minWidth: number, minHeight: number) => void;
  onTranscribe: () => void;
  isLoading: boolean;
  hasImage: boolean;
  hasRegions: boolean;
}

const ControlBar: React.FC<ControlBarProps> = ({
  onImageUpload,
  onBinarize,
  onSegment,
  onTranscribe,
  isLoading,
  hasImage,
  hasRegions,
}) => {
  const [minArea, setMinArea] = useState(500);
  const [minWidth, setMinWidth] = useState(30);
  const [minHeight, setMinHeight] = useState(15);

  const handleUpload = (file: File | UploadFile) => {
    const actualFile = (file as UploadFile).originFileObj || (file as File);
    onImageUpload(actualFile);
    return false;
  };

  return (
    <div className="control-bar">
      <Space size="large" style={{ width: '100%', justifyContent: 'space-between' }}>
        <Space>
          <Upload
            accept="image/*"
            beforeUpload={handleUpload}
            showUploadList={false}
            maxCount={1}
          >
            <Button icon={<UploadOutlined />} size="large" type="primary">
              Upload Image
            </Button>
          </Upload>

          <Button
            icon={<FilterOutlined />}
            size="large"
            onClick={onBinarize}
            disabled={!hasImage || isLoading}
            loading={isLoading}
          >
            Binarize
          </Button>

          <Button
            icon={<ScissorOutlined />}
            size="large"
            onClick={() => onSegment(minArea, minWidth, minHeight)}
            disabled={!hasImage || isLoading}
            loading={isLoading}
          >
            Segment
          </Button>

          <Button
            icon={<FileTextOutlined />}
            size="large"
            type="primary"
            onClick={onTranscribe}
            disabled={!hasRegions || isLoading}
            loading={isLoading}
          >
            Transcribe All
          </Button>
        </Space>

        <Collapse
          size="small"
          items={[
            {
              key: '1',
              label: 'Segmentation Settings',
              children: (
                <Space direction="vertical" style={{ width: 300 }}>
                  <div>
                    <span>Min Area: {minArea}px²</span>
                    <Slider
                      min={100}
                      max={2000}
                      value={minArea}
                      onChange={setMinArea}
                      disabled={isLoading}
                    />
                  </div>
                  <div>
                    <span>Min Width: {minWidth}px</span>
                    <Slider
                      min={10}
                      max={100}
                      value={minWidth}
                      onChange={setMinWidth}
                      disabled={isLoading}
                    />
                  </div>
                  <div>
                    <span>Min Height: {minHeight}px</span>
                    <Slider
                      min={5}
                      max={50}
                      value={minHeight}
                      onChange={setMinHeight}
                      disabled={isLoading}
                    />
                  </div>
                </Space>
              ),
            },
          ]}
        />
      </Space>
    </div>
  );
};

export default ControlBar;