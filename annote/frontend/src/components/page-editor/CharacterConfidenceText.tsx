import type { CharacterConfidence } from './characterConfidence';
import { confidenceHighlightColor, formatConfidencePercent } from './characterConfidence';

type CharacterConfidenceTextProps = {
  characterConfidences: CharacterConfidence[];
  ariaLabel: string;
};

export function CharacterConfidenceText({
  characterConfidences,
  ariaLabel,
}: CharacterConfidenceTextProps) {
  return (
    <div
      aria-label={ariaLabel}
      style={{
        border: '1px solid #3b4350',
        borderRadius: 6,
        padding: '10px 12px',
        background: '#1a1f27',
        lineHeight: 1.8,
        fontFamily: 'Georgia, "Times New Roman", serif',
        fontSize: 18,
        wordBreak: 'break-word',
      }}
    >
      {characterConfidences.map((entry, index) => (
        <span
          key={`${index}-${entry.char}`}
          title={`${entry.char}: ${formatConfidencePercent(entry.confidence)}`}
          style={{
            backgroundColor: confidenceHighlightColor(entry.confidence),
            borderRadius: 2,
            padding: '0 1px',
          }}
        >
          {entry.char}
        </span>
      ))}
    </div>
  );
}
