import { HELPER_BASE_URL } from './constants';
import type { InferenceRunResponse, InferenceTask } from './types';

type RunRequest = {
  task: InferenceTask;
  registry_model_id: string;
  registry_tag?: string;
  image_bytes: string;
  params?: Record<string, unknown>;
  signal?: AbortSignal;
};

export async function runLocalInference(request: RunRequest): Promise<InferenceRunResponse> {
  const response = await fetch(`${HELPER_BASE_URL}/inference/v1/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal: request.signal,
    body: JSON.stringify({
      task: request.task,
      registry_model_id: request.registry_model_id,
      registry_tag: request.registry_tag ?? 'stable',
      image_bytes: request.image_bytes,
      params: request.params ?? {},
    }),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Local inference failed.');
  }
  return (await response.json()) as InferenceRunResponse;
}
