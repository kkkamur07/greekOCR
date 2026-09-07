export const JOB_NOTICE_POLL_INTERVAL_MS = 1500;
// The fallback for when the job stream stalls. A job takes seconds to minutes, so
// four reads a second bought nothing but a query per poll (each one also runs the
// stale-job sweep on the API) against the metered database.
export const JOB_WAIT_POLL_INTERVAL_MS = 1000;
