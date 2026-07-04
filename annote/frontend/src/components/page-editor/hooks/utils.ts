import type {
  LineResponse,
  LineTranscriptionResponse,
  LineUpsertRequest,
  TranscriptionLayerResponse,
} from '../../../api/client';

/** Optional until OpenAPI types include text_source on LineTranscriptionResponse. */
export type LineTranscriptionTextSource = 'model' | 'human_edited';

export type LineTranscriptionWithTextSource = LineTranscriptionResponse & {
  text_source?: LineTranscriptionTextSource | null;
};


export function approvedText(line: LineResponse): string | null {
  return (
    line.line_transcriptions.find(
      (transcription) => transcription.transcription_kind === 'ground_truth',
    )?.text ?? null
  );
}

export function lineTextForLayer(line: LineResponse, transcriptionLayerId: string | null): string {
  if (!transcriptionLayerId) return '';
  return lineTranscriptionForLayer(line, transcriptionLayerId)?.text ?? '';
}

export function lineTranscriptionForLayer(
  line: LineResponse,
  transcriptionLayerId: string | null,
): LineTranscriptionResponse | null {
  if (!transcriptionLayerId) return null;
  return (
    line.line_transcriptions.find(
      (transcription) => transcription.transcription_id === transcriptionLayerId,
    ) ?? null
  );
}

export function modelTranscriptionForLine(line: LineResponse): LineTranscriptionResponse | null {
  return (
    line.line_transcriptions.find(
      (transcription) => transcription.transcription_kind === 'model',
    ) ?? null
  );
}

export function showsModelSourceReview(
  transcription: LineTranscriptionResponse | null,
): boolean {
  if (!transcription) return false;
  if (transcription.transcription_kind === 'model') return true;
  return (transcription as LineTranscriptionWithTextSource).text_source === 'model';
}

export function transcriptionForOcrReview(
  line: LineResponse,
  selectedLayer: TranscriptionLayerResponse | null,
): LineTranscriptionResponse | null {
  if (!selectedLayer) return null;
  const layerTranscription = lineTranscriptionForLayer(line, selectedLayer.id);
  if (selectedLayer.kind === 'model') {
    return layerTranscription;
  }
  if (layerTranscription && showsModelSourceReview(layerTranscription)) {
    return modelTranscriptionForLine(line) ?? layerTranscription;
  }
  return null;
}

export function modelLayerIdForPromotion(
  line: LineResponse,
  selectedLayer: TranscriptionLayerResponse | null,
): string | null {
  if (selectedLayer?.kind === 'model') {
    return selectedLayer.id;
  }
  return modelTranscriptionForLine(line)?.transcription_id ?? null;
}

export function upsertLineRequest(line: LineResponse, order: number): LineUpsertRequest {
  const text = approvedText(line);
  const request: LineUpsertRequest = {
    id: line.id,
    order,
    kind: line.kind,
    points: line.points,
    source: line.source,
  };
  if (text !== null) {
    request.approved_text = text;
  }
  return request;
}

export function withLocalGroundTruth(
  lines: LineResponse[],
  groundTruthTranscriptionId: string | null,
  lineId: string,
  text: string,
): LineResponse[] {
  if (!groundTruthTranscriptionId) return lines;
  return lines.map((line) => {
    if (line.id !== lineId) return line;
    const existing = line.line_transcriptions.filter(
      (transcription) => transcription.transcription_kind !== 'ground_truth',
    );
    const nextTranscription: LineTranscriptionResponse = {
      id:
        line.line_transcriptions.find(
          (transcription) => transcription.transcription_kind === 'ground_truth',
        )?.id ?? `ground-truth-${lineId}`,
      transcription_id: groundTruthTranscriptionId,
      transcription_kind: 'ground_truth',
      text,
      confidence: null,
      text_source: 'human_edited',
      character_confidences: null,
    };
    return { ...line, line_transcriptions: [...existing, nextTranscription] };
  });
}
