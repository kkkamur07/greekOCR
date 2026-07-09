export function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== 'string') {
        reject(new Error('Failed to read image bytes.'));
        return;
      }
      const commaIndex = result.indexOf(',');
      resolve(commaIndex >= 0 ? result.slice(commaIndex + 1) : result);
    };
    reader.onerror = () => reject(reader.error ?? new Error('Failed to read image bytes.'));
    reader.readAsDataURL(blob);
  });
}

export function registrySelectionFromArtifactRef(artifactRef: string): {
  registryModelId: string;
  registryTag: string;
} {
  const match = artifactRef.match(/^registry:\/\/([^/?#]+)(?:\?([^#]*))?/);
  if (!match) {
    throw new Error(`Unsupported artifact_ref: ${artifactRef}`);
  }
  const registryModelId = decodeURIComponent(match[1]);
  const params = new URLSearchParams(match[2] ?? '');
  return {
    registryModelId,
    registryTag: params.get('tag') || 'stable',
  };
}
