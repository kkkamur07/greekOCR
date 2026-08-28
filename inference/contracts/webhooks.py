"""HTTP headers shared between nomikos and the inference runtime.

Only one header, authenticating the platform calling its own job-callback
receiver: an **inference agent** authenticates itself *to* the platform with a
**device credential** instead, and is never called at all (ADR 0002).
"""

INFERENCE_WEBHOOK_SECRET_HEADER = "X-Inference-Webhook-Secret"  # noqa: S105 - a header name
