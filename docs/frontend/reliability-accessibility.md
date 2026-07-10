# Frontend reliability and accessibility

## Cursor pagination

List clients expose `*Page` methods that accept `cursor`, `limit`, and
`AbortSignal`. New list UIs should render the returned page and request the
next cursor only after the user asks for more. `collectCursorPages` remains a
bounded compatibility helper for existing callers: it defaults to ten pages,
rejects repeated cursors, and observes cancellation.

The project jobs panel requests eight jobs at a time and presents a **Load more
jobs** control when the API supplies a next cursor.

## Background job updates

`subscribeToJob` owns one SSE stream per job ID, regardless of how many
components track that job. Subscribers share updates and the owner stops when
the job becomes terminal or its last subscriber leaves.

SSE bytes, including server heartbeat comments, reset a 12-second watchdog. A
stream that cannot open, closes early, or remains silent falls back to
interval polling. This keeps job state current without opening duplicate
streams from the project panel, notices, and editor task tracking.

## Accessible interactions

- Public transcription polygons and editor segment/baseline overlays are
  focusable buttons; Enter and Space perform their selection action.
- Editor vertex controls support pointer drag/removal and keyboard removal.
- `FormModal` moves focus to its first control, traps Tab within the dialog,
  supports Escape, and restores focus to its opener when it closes.
