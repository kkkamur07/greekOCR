export type InferenceTask = 'segment' | 'transcribe' | 'binarize';

export type HostEligibility = 'local' | 'remote' | 'any';

export type CharacterConfidence = {
  char: string;
  confidence: number;
};

export type TranscribeRunOutput = {
  text: string;
  confidence: number;
  character_confidences: CharacterConfidence[];
};

export type TranscribeBatchRunOutput = {
  lines: Array<{
    line_id: string | null;
    line_index: number;
    output: TranscribeRunOutput;
  }>;
};

export type SegmentRunOutput = {
  blocks: Array<Record<string, unknown>>;
  lines: Array<Record<string, unknown>>;
};

export type InferenceRunResponse =
  | { task: 'transcribe'; output: TranscribeRunOutput | TranscribeBatchRunOutput }
  | { task: 'segment'; output: SegmentRunOutput };
