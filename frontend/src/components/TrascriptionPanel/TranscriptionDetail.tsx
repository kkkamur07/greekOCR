import { Button, Input, Space, Typography, Progress, Card, Divider } from 'antd';
import { DeleteOutlined, SaveOutlined, CloseOutlined, CopyOutlined, ReloadOutlined, ArrowLeftOutlined } from '@ant-design/icons';
import { Region, Transcription } from '../../types';
import { useState, useEffect } from 'react';

const { TextArea } = Input;
const { Title, Text, Paragraph } = Typography;

interface TranscriptionDetailProps {
  region: Region;
  transcription?: Transcription;
  onClose: () => void;
  onDelete: () => void;
  onUpdateText: (text: string) => void;
  onTranscribe?: () => void;
}

const TranscriptionDetail: React.FC<TranscriptionDetailProps> = ({
  region,
  transcription,
  onClose,
  onDelete,
  onUpdateText,
  onTranscribe,
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editText, setEditText] = useState('');

  // Sync state when transcription changes
  useEffect(() => {
    setEditText(transcription?.text || '');
  }, [transcription]);

  const handleSave = () => {
    onUpdateText(editText);
    setIsEditing(false);
  };

  const handleCopy = () => {
    if (transcription?.text) {
      navigator.clipboard.writeText(transcription.text);
    }
  };

  const getConfidenceInfo = () => {
    if (!transcription) return { color: '#d9d9d9', percent: 0, label: 'N/A' };
    const pct = transcription.confidence * 100;
    
    let color = '#ff4d4f'; // Default red/error
    if (pct > 90) color = '#52c41a'; // Green
    else if (pct > 70) color = '#1890ff'; // Blue
    else if (pct > 50) color = '#faad14'; // Yellow
    
    return { color, percent: pct, label: `${pct.toFixed(1)}%` };
  };

  const conf = getConfidenceInfo();

  return (
    <div className="transcription-detail">
      {/* Navigation Header */}
      <div className="detail-header">
        <Button 
          type="text" 
          icon={<ArrowLeftOutlined />} 
          onClick={onClose}
          style={{ marginRight: 8 }}
        >
          Back
        </Button>
        <Text strong>Region #{region.id}</Text>
        <div style={{ flex: 1 }} />
        <Button 
          danger 
          type="text" 
          icon={<DeleteOutlined />} 
          onClick={onDelete}
        >
          Delete
        </Button>
      </div>

      <Divider style={{ margin: '0 0 16px 0' }} />

      <div className="detail-content">
        {/* Confidence Section */}
        {transcription && (
          <div className="confidence-section">
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
              <Text type="secondary">Confidence</Text>
              <Text strong style={{ color: conf.color }}>{conf.label}</Text>
            </div>
            <Progress 
              percent={conf.percent} 
              showInfo={false} 
              strokeColor={conf.color} 
              size="small" 
            />
          </div>
        )}

        {/* Text Display / Edit Area */}
        <div className="text-section">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
            <Text type="secondary">Transcription</Text>
            {!isEditing && transcription && (
              <Space>
                <Button size="small" icon={<CopyOutlined />} onClick={handleCopy} />
                <Button size="small" type="primary" ghost icon={<EditOutlined />} onClick={() => setIsEditing(true)}>
                  Edit
                </Button>
              </Space>
            )}
          </div>

          {transcription ? (
            isEditing ? (
              <div className="editor-box">
                <TextArea
                  value={editText}
                  onChange={(e) => setEditText(e.target.value)}
                  autoSize={{ minRows: 6, maxRows: 12 }}
                  className="greek-textarea"
                />
                <Space style={{ marginTop: 12, justifyContent: 'flex-end', width: '100%' }}>
                  <Button onClick={() => setIsEditing(false)}>Cancel</Button>
                  <Button type="primary" icon={<SaveOutlined />} onClick={handleSave}>Save</Button>
                </Space>
              </div>
            ) : (
              <div className="text-display greek-font">
                {transcription.text}
              </div>
            )
          ) : (
            <div className="empty-placeholder">
              <Text type="secondary">No transcription available.</Text>
            </div>
          )}
        </div>

        {/* Actions Footer */}
        <div className="detail-actions">
          {onTranscribe && (
            <Button
              block
              size="large"
              icon={transcription ? <ReloadOutlined /> : <ThunderboltOutlined />}
              onClick={onTranscribe}
              className={transcription ? "retranscribe-btn" : "transcribe-btn-primary"}
            >
              {transcription ? "Re-transcribe Region" : "Transcribe Region"}
            </Button>
          )}
          
          <div style={{ marginTop: 16, textAlign: 'center' }}>
            <Text type="secondary" style={{ fontSize: 12 }}>
              Pixel Coordinates: X:{Math.round(region.bbox[0])}, Y:{Math.round(region.bbox[1])}
            </Text>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TranscriptionDetail;
