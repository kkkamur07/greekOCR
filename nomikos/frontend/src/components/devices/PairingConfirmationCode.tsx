import type { CSSProperties } from "react";

/**
 * The confirmation code, at the size the decision deserves.
 *
 * A forwarded consent link mints a token on whoever clicks it, for a computer
 * that is not theirs (ADR 0001, "Pairing phishing"). Comparing this code with
 * the one the helper printed is the only thing standing in the way, so it is
 * not a detail row: it is the largest thing on the screen, and the copy says
 * what to do with it rather than merely labelling it.
 *
 * The styles are inline because the shared stylesheet has no vocabulary for a
 * one-off display of this kind; they are written against the same tokens the
 * sheet defines.
 */
const frameStyle: CSSProperties = {
  border: "1px solid var(--border-2)",
  borderRadius: "var(--radius-lg)",
  background: "var(--surface-2)",
  padding: "var(--space-5)",
  margin: "var(--space-5) 0",
  textAlign: "center",
};

const codeStyle: CSSProperties = {
  display: "block",
  fontFamily:
    'ui-monospace, SFMono-Regular, Menlo, Consolas, "Courier New", monospace',
  fontSize: "2.25rem",
  fontWeight: 700,
  lineHeight: 1.2,
  color: "var(--navy)",
  // Tracking this wide leaves a phantom space after the last glyph; the extra
  // left padding puts the run back on the centre line.
  letterSpacing: "0.3em",
  paddingLeft: "0.3em",
  wordBreak: "break-all",
};

const captionStyle: CSSProperties = {
  marginTop: "var(--space-3)",
  fontSize: "0.8125rem",
  color: "var(--text-3)",
};

export function PairingConfirmationCode({ code }: { code: string }) {
  return (
    <div style={frameStyle} role="group" aria-labelledby="pair-code-label">
      <p className="section-label" style={{ margin: 0 }} id="pair-code-label">
        Confirmation code
      </p>
      {/*
        Announced one character at a time. Read as a word, "K7QF-2M9X" is a
        noise a researcher cannot check against their terminal, which is the
        only thing this code is for. It is a second copy rather than an
        `aria-label`, because ARIA prohibits naming a `strong`.
      */}
      <strong style={codeStyle} aria-hidden="true">
        {code}
      </strong>
      <span className="visually-hidden">{code.split("").join(" ")}</span>
      <p style={captionStyle}>
        Your terminal printed this code. If it does not match, deny the request
        - someone else asked for this.
      </p>
    </div>
  );
}
