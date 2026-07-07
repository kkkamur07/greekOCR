import type { PublicLineResponse } from '../api/client';
import type { Region, Transcription } from '../types';

export function linesForPart(
  lines: PublicLineResponse[] | undefined,
  partId: string,
): PublicLineResponse[] {
  return [...(lines ?? [])].filter((line) => line.part_id === partId).sort((a, b) => a.order - b.order);
}

export function publicLinesToRegions(lines: PublicLineResponse[]): Region[] {
  return lines.map((line, index) => {
    const points = line.points.map((point) => [point[0], point[1]] as [number, number]);
    const xs = points.map((point) => point[0]);
    const ys = points.map((point) => point[1]);
    return {
      id: index + 1,
      boundary: points,
      bbox: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)],
    };
  });
}

export function lineTextForLayer(
  line: PublicLineResponse,
  layerId: string | null,
): string | null {
  const transcriptions = line.line_transcriptions ?? [];
  const match = layerId
    ? transcriptions.find((item) => item.transcription_id === layerId)
    : transcriptions.find((item) => item.transcription_kind === 'ground_truth');
  const picked = match ?? transcriptions[0];
  const text = picked?.text?.trim();
  return text ? text : null;
}

export function publicLinesToTranscriptions(
  lines: PublicLineResponse[],
  layerId: string | null,
): Transcription[] {
  return lines.flatMap((line, index) => {
    const text = lineTextForLayer(line, layerId);
    if (!text) return [];
    const confidence =
      line.line_transcriptions?.find((item) => item.text === text)?.confidence ?? 1;
    return [{ region_id: index + 1, text, confidence: confidence ?? 1 }];
  });
}
