/**
 * A thing the editor has to say, and which saying it is.
 *
 * The editor's success feedback is a toast raised from an effect keyed on the
 * message. Several of those messages are constants (e.g. "Ground truth text
 * saved", "Saved to Ground truth", "Layout reset"), so saving twice in a row
 * produced the same string twice: React saw an unchanged dependency and the
 * second save showed nothing. They're toast-only with no sticky line behind
 * them, so there was nothing else on screen to notice either.
 *
 * The token is what makes the second save a *new* message rather than the
 * same one. It's not a timestamp anyone reads; the only property that
 * matters is that two consecutive messages never share it.
 */
export type StatusMessage = {
  text: string;
  /** Strictly increasing across the session. See `statusMessage`. */
  at: number;
};

let lastToken = 0;

/** Stamps `text` with a token no earlier message carries. */
export function statusMessage(text: string): StatusMessage {
  // `Date.now()` repeats inside a millisecond, which is exactly the case this
  // exists for, so the counter never stands still.
  lastToken = Math.max(lastToken + 1, Date.now());
  return { text, at: lastToken };
}
