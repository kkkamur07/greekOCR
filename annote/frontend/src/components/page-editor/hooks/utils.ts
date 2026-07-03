import type {
  LineResponse,
  LineTranscriptionResponse,
  LineUpsertRequest,
} from '../../../api/client';

export function approvedText(line: LineResponse): string | null {
  return (
    line.line_transcriptions.find(
      (transcription) => transcription.transcription_kind === 'ground_truth',
    )?.text ?? null
  );
}

export function lineTextForLayer(line: LineResponse, transcriptionLayerId: string | null): string {
  if (!transcriptionLayerId) return '';
  return (
    line.line_transcriptions.find(
      (transcription) => transcription.transcription_id === transcriptionLayerId,
    )?.text ?? ''
  );
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
    };
    return { ...line, line_transcriptions: [...existing, nextTranscription] };
  });
}
