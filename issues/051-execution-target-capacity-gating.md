---
id: "051"
title: "execution-target-capacity-gating"
type: AFK
status: done
tracker: "https://github.com/kkkamur07/greekOCR/issues/51"
blocked_by:
  - "048-collapse-second-job-queue.md"
parent_prd: "https://github.com/kkkamur07/greekOCR/issues/47"
---

## Parent

#47

## What to build

Give jobs an **execution target** — `local` or `cloud` — fixed at submission and never changed afterwards, and gate submission on **capacity**.

**Capacity** is whether an **inference host** currently has a machine able to take work, answered by whether any device for that host was seen recently. The researcher's laptop and a hosted worker are the same kind of thing here, so cloud availability is not a separate concept.

Selection is an account-level setting ("use my computer when it is available") plus implicit choice. There is no per-job toggle: a researcher cannot know which host is faster for a given page, so asking at every action is a decision without a basis.

The researcher is always told which host will run the job, and a job that fails reports which host it failed on. When the preferred host has no **capacity**, the job goes to the other host *and says so* — never silently. When neither host has **capacity**, submission refuses with a clear reason rather than creating a job nobody will claim. Cloud **capacity** will not exist for some time and this requires no special handling.

There is no cloud-fallback timer, no hold window, and no sweeper for unclaimed local jobs. The decision is made once, before the job exists.

Remove `local_only` as a target. Its justification — manuscripts never leave the machine — was never true, since page images already live in the platform's media store, and it was the one mode that could leave a job with no terminal outcome.

**Host eligibility** constrains which targets a job may choose; it does not choose one.

Per ADR 0002.

## Acceptance criteria

- [ ] Jobs carry an **execution target** set at submission and rejected on any attempt to change it
- [ ] Submission consults **capacity** per host from recent device activity
- [ ] Submission with preferred host unavailable but the other available succeeds and reports the substitution
- [ ] Submission with no **capacity** on either host is refused with a reason naming the situation
- [ ] A **registry model id** that is not a **lite model tier** is ineligible for `local`
- [ ] A failed job reports which **inference host** it failed on
- [ ] `local_only` is absent from the schema, the API, and the database
- [ ] Tested over HTTP against the real application factory with live Postgres

## Blocked by

- #48
