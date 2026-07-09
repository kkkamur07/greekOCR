"""Mockable Hugging Face Hub download client."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class HubClient(Protocol):
  def resolve_revision_sha(self, repo_id: str, revision: str) -> str: ...

  def snapshot_download(self, repo_id: str, revision: str, local_dir: Path) -> None: ...


def _hub_error_message(exc: Exception, *, repo_id: str, revision: str) -> str:
  name = type(exc).__name__
  message = str(exc).strip()

  if name == "RepositoryNotFoundError":
    return (
      f"Hub model repo not found: {repo_id}. "
      "Check the namespace and hub repo slug in the registry weights source."
    )
  if name == "RevisionNotFoundError":
    return (
      f"Hub revision not found: {repo_id}@{revision}. "
      "Check the registry tag maps to an existing git tag or commit on the Hub repo."
    )
  if name in {"GatedRepoError", "HfHubHTTPError"} and (
    "401" in message or "403" in message or "authorized" in message.lower()
  ):
    return (
      f"Hub access denied for {repo_id}@{revision}. "
      "Set HF_TOKEN only for private or gated repos; public nomicous repos do not require a token."
    )
  if name == "OfflineModeIsEnabled":
    return (
      f"Hub is offline and {repo_id}@{revision} is not cached. "
      "Run scripts/hf/fetch_model.py while online or use local bundled weights."
    )

  return (
    f"Hub download failed for {repo_id}@{revision}: {message or name}. "
    "Verify the repo is public, the revision exists, and network access is available."
  )


class HuggingFaceHubClient:
  def resolve_revision_sha(self, repo_id: str, revision: str) -> str:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    try:
      info = HfApi().repo_info(repo_id=repo_id, revision=revision, repo_type="model")
    except Exception as exc:
      if isinstance(exc, HfHubHTTPError) and exc.response is not None:
        if exc.response.status_code in {401, 403}:
          raise ValueError(_hub_error_message(exc, repo_id=repo_id, revision=revision)) from exc
      if type(exc).__name__ in {
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
        "GatedRepoError",
        "OfflineModeIsEnabled",
      }:
        raise ValueError(_hub_error_message(exc, repo_id=repo_id, revision=revision)) from exc
      raise ValueError(_hub_error_message(exc, repo_id=repo_id, revision=revision)) from exc

    sha = getattr(info, "sha", None)
    if not sha:
      raise ValueError(f"could not resolve Hub revision for {repo_id}@{revision}")
    return str(sha)

  def snapshot_download(self, repo_id: str, revision: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    try:
      local_dir.mkdir(parents=True, exist_ok=True)
      snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(local_dir),
        repo_type="model",
      )
    except Exception as exc:
      raise ValueError(_hub_error_message(exc, repo_id=repo_id, revision=revision)) from exc


_default_client: HubClient | None = None


def get_default_hub_client() -> HubClient:
  global _default_client
  if _default_client is None:
    _default_client = HuggingFaceHubClient()
  return _default_client


def set_default_hub_client(client: HubClient | None) -> None:
  global _default_client
  _default_client = client
