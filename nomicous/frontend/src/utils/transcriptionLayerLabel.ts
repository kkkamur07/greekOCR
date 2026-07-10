import type { TranscriptionLayerResponse } from "../api/client";

export function groundTruthLayer(
  layers: TranscriptionLayerResponse[],
): TranscriptionLayerResponse | null {
  return layers.find((layer) => layer.kind === "ground_truth") ?? null;
}
