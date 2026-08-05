// The helper serves models from the platform's registry and echoes its
// per-character score shape, so both vocabularies come from the generated
// schema rather than a second copy here.
import type { CharacterConfidence, InferenceTask } from "../api/client";

export type { CharacterConfidence, InferenceTask };

/** Helper-only: the platform catalog does not publish routing eligibility. */
export type HostEligibility = "local" | "remote" | "any";

export type TranscribeRunOutput = {
  text: string;
  confidence: number;
  character_confidences: CharacterConfidence[];
};

export type TranscribeBatchRunOutput = {
  lines: Array<{
    line_id: string | null;
    line_index: number;
    // Exactly one of these is set. The inference service isolates per-line
    // failures rather than discarding the whole page, so a batch can come back
    // as a partial success and `output` is absent on the lines that failed.
    output: TranscribeRunOutput | null;
    error?: string | null;
  }>;
};

export type SegmentRunOutput = {
  blocks: Array<Record<string, unknown>>;
  lines: Array<Record<string, unknown>>;
};

export type InferenceRunResponse =
  | {
      task: "transcribe";
      output: TranscribeRunOutput | TranscribeBatchRunOutput;
    }
  | { task: "segment"; output: SegmentRunOutput };
