const STORAGE_KEY = "nomicous_inference_preference";

export type InferencePreference = "local" | "cloud";

export function loadInferencePreference(): InferencePreference {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw === "cloud" ? "cloud" : "local";
  } catch {
    return "local";
  }
}

export function saveInferencePreference(preference: InferencePreference): void {
  localStorage.setItem(STORAGE_KEY, preference);
}

export function preferCloudInference(): boolean {
  return loadInferencePreference() === "cloud";
}
