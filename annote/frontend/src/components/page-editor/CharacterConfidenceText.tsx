import type { CharacterConfidence } from './characterConfidence';
import {
  confidenceTierClass,
  confidenceTierLabel,
  formatConfidencePercent,
} from './characterConfidence';

type CharacterConfidenceTextProps = {
  characterConfidences: CharacterConfidence[];
  ariaLabel: string;
};

export function CharacterConfidenceText({
  characterConfidences,
  ariaLabel,
}: CharacterConfidenceTextProps) {
  return (
    <span className="pe-confidence-text" aria-label={ariaLabel}>
      {characterConfidences.map((entry, index) => {
        const tier = confidenceTierLabel(entry.confidence);
        const pct = formatConfidencePercent(entry.confidence);
        return (
          <span
            key={`${index}-${entry.char}`}
            className={confidenceTierClass(entry.confidence)}
            data-conf={Math.round(entry.confidence * 100)}
            data-tier={tier}
            title={`${pct} confidence (${tier})`}
          >
            {entry.char}
          </span>
        );
      })}
    </span>
  );
}
