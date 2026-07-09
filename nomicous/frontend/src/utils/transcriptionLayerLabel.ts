import type { TranscriptionLayerResponse } from '../api/client';

function modelLayers(layers: TranscriptionLayerResponse[]): TranscriptionLayerResponse[] {
  return layers.filter((layer) => layer.kind === 'model');
}

export function formatTranscriptionLayerLabel(
  layer: TranscriptionLayerResponse,
  layers: TranscriptionLayerResponse[],
): string {
  if (layer.kind === 'ground_truth') {
    return 'Ground truth';
  }

  const models = modelLayers(layers);
  if (models.length <= 1) {
    return 'Model output';
  }

  const index = models.findIndex((item) => item.id === layer.id);
  const runNumber = index >= 0 ? models.length - index : models.length;
  return `Model output #${runNumber}`;
}

export function groundTruthLayer(
  layers: TranscriptionLayerResponse[],
): TranscriptionLayerResponse | null {
  return layers.find((layer) => layer.kind === 'ground_truth') ?? null;
}
