# Architecture Decision Records

One file per decision, named `NNNN-kebab-case-title.md`, numbered in the order the
decision was taken. A record is written when a choice is expensive to reverse:
schema shape, credential design, transport direction, trust boundaries.

Records are immutable once merged. A decision that is later reversed gets a new
record that supersedes the old one, and the old one gains a `Superseded by` line.
Do not edit history — the value of an ADR is that it says what was believed at
the time, and why.

| # | Title | Status |
|---|-------|--------|
| [0001](0001-outbound-helper-device-pairing.md) | Outbound helper device pairing and device-scoped tokens | Accepted |
