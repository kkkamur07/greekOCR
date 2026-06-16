import { useState } from 'react';
import { Layout, message } from 'antd';
import ImageCanvas from './components/ImageCanvas/ImageCanvas';
import TranscriptionPanel from './components/TrascriptionPanel/TrascriptionPanel';
import ControlBar from './components/ControlBar/ControlBar';
import { Region, Transcription } from './types';
import { useHistory } from './hooks/useHistory';
import {
  uploadImage,
  segmentImage,
  transcribeRegions,
  transcribeSingleRegion,
  getImageUrl,
  binarizeImage,
} from './services/api';
import './App.css';

const { Header, Content } = Layout;

export default function LegacyDemoApp() {
  const [imageId, setImageId] = useState<string | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
  const [isLoading, setIsLoading] = useState(false);

  const history = useHistory();
  const { currentState, pushState, undo, redo, canUndo, canRedo } = history;
  const { regions, transcriptions, selectedRegionId } = currentState;

  const updateState = (
    newRegions: Region[],
    newTranscriptions: Transcription[],
    newSelectedId: number | null,
  ) => {
    pushState(newRegions, newTranscriptions, newSelectedId);
  };

  const handleImageUpload = async (file: File) => {
    setIsLoading(true);
    try {
      const response = await uploadImage(file);
      setImageId(response.image_id);
      setImageUrl(getImageUrl(response.image_id));
      setImageDimensions({ width: response.width, height: response.height });
      updateState([], [], null);
      message.success('Image uploaded successfully!');
    } catch (error) {
      message.error('Failed to upload image');
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleBinarize = async () => {
    if (!imageId) {
      message.warning('Please upload an image first');
      return;
    }

    setIsLoading(true);
    try {
      const response = await binarizeImage(imageId);
      setImageId(response.image_id);
      setImageUrl(getImageUrl(response.image_id));
      updateState([], [], null);
      message.success('Image binarized successfully!');
    } catch (error) {
      message.error('Failed to binarize image');
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSegment = async (minArea: number, minWidth: number, minHeight: number) => {
    if (!imageId) {
      message.warning('Please upload an image first');
      return;
    }

    setIsLoading(true);
    try {
      const response = await segmentImage(imageId, minArea, minWidth, minHeight);
      updateState(response.regions, [], null);
      message.success(`Found ${response.total_regions} regions`);
    } catch (error) {
      message.error('Failed to segment image');
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleTranscribe = async () => {
    if (!imageId || regions.length === 0) {
      message.warning('Please segment the image first');
      return;
    }

    setIsLoading(true);
    try {
      const response = await transcribeRegions(imageId, regions);
      updateState(regions, response.transcriptions, selectedRegionId);
      message.success(`Transcribed ${response.transcriptions.length} regions`);
    } catch (error) {
      message.error('Failed to transcribe regions');
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleTranscribeRegion = async (regionId: number) => {
    if (!imageId) return;

    const region = regions.find((r) => r.id === regionId);
    if (!region) return;

    setIsLoading(true);
    try {
      const response = await transcribeSingleRegion(imageId, region);
      const filtered = transcriptions.filter((t) => t.region_id !== regionId);
      const newTranscriptions = [...filtered, ...response.transcriptions];
      updateState(regions, newTranscriptions, selectedRegionId);
      message.success(`Transcribed region #${regionId}`);
    } catch (error) {
      message.error(`Failed to transcribe region #${regionId}`);
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAddRegion = (region: Region) => {
    updateState([...regions, region], transcriptions, selectedRegionId);
  };

  const handleUpdateRegion = (updatedRegion: Region) => {
    const newRegions = regions.map((r) => (r.id === updatedRegion.id ? updatedRegion : r));
    const newTranscriptions = transcriptions.filter((t) => t.region_id !== updatedRegion.id);
    updateState(newRegions, newTranscriptions, selectedRegionId);
  };

  const handleDeleteRegion = (regionId: number) => {
    const newRegions = regions.filter((r) => r.id !== regionId);
    const newTranscriptions = transcriptions.filter((t) => t.region_id !== regionId);
    const newSelectedId = selectedRegionId === regionId ? null : selectedRegionId;
    updateState(newRegions, newTranscriptions, newSelectedId);
  };

  const handleUpdateTranscription = (regionId: number, newText: string) => {
    const newTranscriptions = transcriptions.map((t) =>
      t.region_id === regionId ? { ...t, text: newText } : t,
    );
    updateState(regions, newTranscriptions, selectedRegionId);
  };

  const handleSelectRegion = (id: number | null) => {
    updateState(regions, transcriptions, id !== null ? Number(id) : null);
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header
        style={{
          background: '#001529',
          padding: '0 24px',
          display: 'flex',
          alignItems: 'center',
          color: 'white',
        }}
      >
        <h1 style={{ color: 'white', margin: 0, fontSize: '24px' }}>
          Greek Manuscript OCR (legacy demo)
        </h1>
      </Header>

      <ControlBar
        onImageUpload={handleImageUpload}
        onBinarize={handleBinarize}
        onSegment={handleSegment}
        onTranscribe={handleTranscribe}
        isLoading={isLoading}
        hasImage={!!imageId}
        hasRegions={regions.length > 0}
      />

      <Content style={{ padding: '24px', background: '#f0f2f5' }}>
        <div style={{ display: 'flex', gap: '24px', height: 'calc(100vh - 180px)' }}>
          <div
            style={{ flex: '0 0 65%', background: 'white', borderRadius: '8px', overflow: 'hidden' }}
          >
            <ImageCanvas
              imageUrl={imageUrl}
              imageDimensions={imageDimensions}
              regions={regions}
              selectedRegionId={selectedRegionId}
              onSelectRegion={handleSelectRegion}
              onAddRegion={handleAddRegion}
              onUpdateRegion={handleUpdateRegion}
              onDeleteRegion={handleDeleteRegion}
              onTranscribeRegion={handleTranscribeRegion}
              onUndo={undo}
              onRedo={redo}
              canUndo={canUndo}
              canRedo={canRedo}
            />
          </div>

          <div
            style={{ flex: '0 0 35%', background: 'white', borderRadius: '8px', overflow: 'hidden' }}
          >
            <TranscriptionPanel
              regions={regions}
              transcriptions={transcriptions}
              selectedRegionId={selectedRegionId}
              onSelectRegion={handleSelectRegion}
              onUpdateTranscription={handleUpdateTranscription}
              onDeleteRegion={handleDeleteRegion}
              onTranscribeRegion={handleTranscribeRegion}
            />
          </div>
        </div>
      </Content>
    </Layout>
  );
}
