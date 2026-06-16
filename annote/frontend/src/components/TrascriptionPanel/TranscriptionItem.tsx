import { Typography } from 'antd';
import { Region, Transcription } from '../../types';

const { Text } = Typography;

interface TranscriptionItemProps {
  region: Region;
  transcription?: Transcription;
  isSelected: boolean;
  onSelect: () => void;
}

// List Item View (Compact only)
const TranscriptionItem: React.FC<TranscriptionItemProps> = ({
  region,
  transcription,
  isSelected,
  onSelect,
  // onDelete, // Not used in compact view mostly, or minimal
}) => {
  const getStatusColor = () => {
    if (!transcription) return '#d9d9d9'; // Grey
    if (transcription.confidence > 0.9) return '#52c41a'; // Green
    if (transcription.confidence > 0.7) return '#1890ff'; // Blue
    return '#faad14'; // Orange
  };

  return (
    <div
      className={`transcription-item compact ${isSelected ? 'selected' : ''}`}
      onClick={onSelect}
    >
      <div className="indicator" style={{ background: getStatusColor() }} />

      <div className="compact-content">
        <div className="compact-header">
          <Text strong className="region-id">#{region.id}</Text>
          {transcription && (
            <Text type="secondary" style={{ fontSize: 11 }}>
              {(transcription.confidence * 100).toFixed(0)}%
            </Text>
          )}
        </div>
        <div className="compact-preview">
          {transcription ? (
             <Text ellipsis className="greek-font-preview">{transcription.text}</Text>
          ) : (
             <Text type="secondary" italic>Not described</Text>
          )}
        </div>
      </div>
    </div>
  );
};

export default TranscriptionItem;