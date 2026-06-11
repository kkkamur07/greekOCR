import type { Segment } from "@/types/api";

function confidenceClass(probability: number): string {
  if (probability >= 0.9) return "bg-emerald-200/90 text-emerald-950";
  if (probability >= 0.7) return "bg-amber-200/90 text-amber-950";
  return "bg-rose-200/90 text-rose-950";
}

function formatConfidence(probability: number): string {
  return `${Math.round(probability * 100)}%`;
}

interface ModelSuggestionProps {
  text: string;
  confidence?: Segment["model_transcription_confidence"];
}

export default function ModelSuggestion({ text, confidence }: ModelSuggestionProps) {
  const chars = Array.from(text);
  const showConfidence = confidence != null && confidence.length > 0;

  return (
    <div className="rounded-lg border border-indigo-100 bg-indigo-50/50 px-3 py-2 text-sm text-indigo-950">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs font-medium text-indigo-700">Model suggestion</p>
        {showConfidence && (
          <p className="text-[11px] text-indigo-600">Hover a character for confidence</p>
        )}
      </div>
      {showConfidence ? (
        <p className="mt-1 whitespace-pre-wrap leading-relaxed">
          {chars.map((char, index) => {
            const item = confidence[index];
            if (!item) {
              return <span key={`${index}-${char}`}>{char}</span>;
            }
            return (
              <span
                key={`${index}-${char}-${item.probability}`}
                title={`${item.char || char}: ${formatConfidence(item.probability)}`}
                className={`rounded-sm px-0.5 ${confidenceClass(item.probability)}`}
              >
                {char}
              </span>
            );
          })}
        </p>
      ) : (
        <p className="mt-0.5 whitespace-pre-wrap leading-relaxed">{text}</p>
      )}
    </div>
  );
}
