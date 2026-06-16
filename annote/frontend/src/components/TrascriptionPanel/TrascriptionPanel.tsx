import { List, Input, Button, Empty, Progress, Space, Typography } from 'antd';
import { DeleteOutlined, EditOutlined } from '@ant-design/icons';
import { Region, Transcription } from '../../types';
import { useState } from 'react';
import './TrascriptionPanel.css';

const { TextArea } = Input;
const { Text } = Typography;

interface TranscriptionPanelProps {
  regions: Region[];
  transcriptions: Transcription[];
  selectedRegionId: number | null;
  onSelectRegion: (id: number) => void;
  onUpdateTranscription: (regionId: number, text: string) => void;
  onDeleteRegion: (regionId: number) => void;
}

const TranscriptionPanel: React.FC<TranscriptionPanelProps> = ({
  regions,
  transcriptions,
  selectedRegionId,
  onSelectRegion,
  onUpdateTranscription,
  onDeleteRegion,
}) => {
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editText, setEditText] = useState('');

  const getTranscription = (regionId: number) => {
    return transcriptions.find(t => t.region_id === regionId);
  };

  const handleEdit = (regionId: number, currentText: string) => {
    setEditingId(regionId);
    setEditText(currentText);
  };

  const handleSave = (regionId: number) => {
    onUpdateTranscription(regionId, editText);
    setEditingId(null);
  };

  return (
    <div className="transcription-panel">
      <div className="panel-header">
        <h3>Transcriptions</h3>
        <Text type="secondary">Total: {regions.length}</Text>
      </div>

      <div className="panel-content">
        {regions.length === 0 ? (
          <Empty description="No regions detected" />
        ) : (
          <List
            dataSource={regions}
            renderItem={(region) => {
              const transcription = getTranscription(region.id);
              const isSelected = selectedRegionId === region.id;
              const isEditing = editingId === region.id;

              return (
                <List.Item
                  className={`transcription-item ${isSelected ? 'selected' : ''}`}
                  onClick={() => onSelectRegion(region.id)}
                >
                  <div className="item-content">
                    <div className="item-header">
                      <Text strong style={{ color: isSelected ? '#1890ff' : '#000' }}>
                        Region #{region.id}
                      </Text>
                      <Space>
                        {transcription && !isEditing && (
                          <Button
                            size="small"
                            icon={<EditOutlined />}
                            onClick={(e) => {
                              e.stopPropagation();
                              handleEdit(region.id, transcription.text);
                            }}
                          />
                        )}
                        <Button
                          size="small"
                          danger
                          icon={<DeleteOutlined />}
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteRegion(region.id);
                          }}
                        />
                      </Space>
                    </div>

                    {transcription ? (
                      <>
                        {isEditing ? (
                          <div onClick={(e) => e.stopPropagation()}>
                            <TextArea
                              value={editText}
                              onChange={(e) => setEditText(e.target.value)}
                              autoSize={{ minRows: 2, maxRows: 6 }}
                              style={{ marginTop: 8 }}
                            />
                            <Space style={{ marginTop: 8 }}>
                              <Button size="small" type="primary" onClick={() => handleSave(region.id)}>
                                Save
                              </Button>
                              <Button size="small" onClick={() => setEditingId(null)}>
                                Cancel
                              </Button>
                            </Space>
                          </div>
                        ) : (
                          <>
                            <div className="transcription-text">
                              {transcription.text}
                            </div>
                            <div className="confidence-bar">
                              <Text type="secondary" style={{ fontSize: 12 }}>
                                Confidence: {(transcription.confidence * 100).toFixed(1)}%
                              </Text>
                              <Progress
                                percent={transcription.confidence * 100}
                                size="small"
                                status={transcription.confidence > 0.8 ? 'success' : 'normal'}
                                showInfo={false}
                              />
                            </div>
                          </>
                        )}
                      </>
                    ) : (
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        Not yet transcribed
                      </Text>
                    )}

                    <Text type="secondary" style={{ fontSize: 11 }}>
                      Position: ({region.bbox[0]}, {region.bbox[1]}) |
                      Size: {region.bbox[2] - region.bbox[0]} × {region.bbox[3] - region.bbox[1]}px
                    </Text>
                  </div>
                </List.Item>
              );
            }}
          />
        )}
      </div>
    </div>
  );
};

export default TranscriptionPanel;