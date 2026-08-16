"""Errors the media store protocol is allowed to raise."""


class PresignUnsupported(Exception):
    """This backend cannot mint presigned upload URLs.

    Distinct from ``ValueError`` on purpose: the Supabase store raises
    ``ValueError`` for a malformed object key, and a caller that treated every
    ``ValueError`` as "backend cannot presign" would silently re-route a
    key-validation bug into the multipart fallback path.
    """
