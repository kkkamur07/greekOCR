"""HTTP headers shared between nomicous and the inference runtime.

One header, and it names the direction that survives: the platform calling its
own job-callback receiver. ``X-Inference-Service-Secret`` authenticated callers
of the loopback service's ``POST /inference/v1/run`` and died with it (ADR
0002) - an **inference agent** now authenticates itself *to* the platform with a
**device credential**, and is never called at all.
"""

INFERENCE_WEBHOOK_SECRET_HEADER = "X-Inference-Webhook-Secret"  # noqa: S105 - a header name
