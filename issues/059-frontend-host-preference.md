---
id: "059"
title: "frontend-host-preference"
type: AFK
status: in_progress
tracker: "https://github.com/kkkamur07/greekOCR/issues/59"
blocked_by:
  - "051-execution-target-capacity-gating.md"
parent_prd: "https://github.com/kkkamur07/greekOCR/issues/47"
---

## Parent

#47

## What to build

The researcher-facing half of **execution target**: an account-level setting, and an announcement on every job.

The setting is "use my computer when it is available", chosen once at the account level. There is no per-job toggle — see #51 for why.

The announcement line is **not cosmetic. It is the entire user interface for this feature.** It belongs on the job rather than in a toast, because a researcher who looks away must still be able to read where their job went. It says which **inference host** will run the job, and when the preferred host was unavailable it says that plainly rather than substituting in silence.

Remove `local_only` from the picker, and with it the copy claiming nothing is sent to the cloud — which is already false, since page images live in the platform's media store and the browser downloads them from there today.

## Acceptance criteria

- [ ] An account-level setting expresses the local preference and persists
- [ ] Every job displays which **inference host** will run it
- [ ] A substituted host is stated plainly on the job, not in a transient toast
- [ ] A refused submission explains that no host had **capacity**
- [ ] A failed job shows which host it failed on
- [ ] `local_only` and its privacy copy are gone from the interface
- [ ] No per-job execution target control exists

## Blocked by

- #51
