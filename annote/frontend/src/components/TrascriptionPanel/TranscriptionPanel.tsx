import { Empty, Typography, Space } from 'antd';
import { Region, Transcription } from '../../types';
import TranscriptionItem from './TranscriptionItem';
import TranscriptionDetail from './TranscriptionDetail';
import './TrascriptionPanel.css';

const { Text } = Typography;

interface TranscriptionPanelProps {
  regions: Region[];
  transcriptions: Transcription[];
  selectedRegionId: number | null;
  onSelectRegion: (id: number | null) => void;
  onUpdateTranscription: (regionId: number, text: string) => void;
  onDeleteRegion: (regionId: number) => void;
  onTranscribeRegion?: (regionId: number) => void;
  readOnly?: boolean;
}

const TranscriptionPanel: React.FC<TranscriptionPanelProps> = ({
  regions,
  transcriptions,
  selectedRegionId,
  onSelectRegion,
  onUpdateTranscription,
  onDeleteRegion,
  onTranscribeRegion,
  readOnly = false,
}) => {
  const getTranscription = (regionId: number) => {
    return transcriptions.find(t => t.region_id === regionId);
  };

  // Robust selection finding: Handle potential string/number mismatches
  const safeSelectedId = selectedRegionId !== null ? Number(selectedRegionId) : null;
  const selectedRegion = safeSelectedId !== null
    ? regions.find(r => Number(r.id) === safeSelectedId)
    : null;

  // Detail View Mode happens when a region is selected
  if (safeSelectedId !== null && selectedRegion) {
    return (
      <div className="transcription-panel detail-mode">
        <TranscriptionDetail
          region={selectedRegion}
          transcription={getTranscription(safeSelectedId)}
          readOnly={readOnly}
          onClose={() => onSelectRegion(null)}
          onDelete={() => onDeleteRegion(safeSelectedId)}
          onUpdateText={(text) => onUpdateTranscription(safeSelectedId, text)}
          onTranscribe={
            readOnly || !onTranscribeRegion
              ? undefined
              : () => onTranscribeRegion(safeSelectedId)
          }
        />
      </div>
    );
  }

  // Default List Mode
  return (
    <div className="transcription-panel">
      <div className="panel-header">
        <h3>Transcriptions</h3>
        <Space size="large">
          <Text type="secondary">Total: {regions.length}</Text>
          <Text type="secondary">Transcribed: {transcriptions.length}</Text>
        </Space>
      </div>

      <div className="panel-content">
        {regions.length === 0 ? (
          <Empty description="No regions detected" image={Empty.PRESENTED_IMAGE_SIMPLE} />
        ) : (
          regions.map((region) => (
            <TranscriptionItem
              key={region.id}
              region={region}
              transcription={getTranscription(region.id)}
              isSelected={selectedRegionId === region.id}
              onSelect={() => onSelectRegion(region.id)}
            />
          ))
        )}
      </div>
    </div>
  );
};

export default TranscriptionPanel;