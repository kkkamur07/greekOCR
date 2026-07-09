"""Mockable Hugging Face Hub publish client."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class PublishClient(Protocol):
  def create_repo(
    self,
    repo_id: str,
    *,
    repo_type: str,
    private: bool = False,
  ) -> None: ...

  def upload_folder(
    self,
    folder_path: Path,
    repo_id: str,
    *,
    repo_type: str,
    commit_message: str,
  ) -> str: ...

  def create_tag(
    self,
    repo_id: str,
    *,
    tag: str,
    revision: str,
    repo_type: str,
  ) -> None: ...

  def update_collection_metadata(
    self,
    collection_slug: str,
    *,
    title: str | None = None,
    description: str | None = None,
  ) -> None: ...

  def add_collection_item(
    self,
    collection_slug: str,
    *,
    item_id: str,
    item_type: str,
    note: str | None = None,
  ) -> None: ...

  def get_collection(self, collection_slug: str): ...


@dataclass
class MockPublishClient:
  repos: list[tuple[str, str, bool]] = field(default_factory=list)
  uploads: list[tuple[Path, str, str, str]] = field(default_factory=list)
  tags: list[tuple[str, str, str, str]] = field(default_factory=list)
  collection_updates: list[tuple[str, str | None, str | None]] = field(
    default_factory=list
  )
  collection_items: list[tuple[str, str, str, str | None]] = field(
    default_factory=list
  )
  collections: dict[str, object] = field(default_factory=dict)
  upload_revision: str = "mock-commit-sha"

  def create_repo(
    self,
    repo_id: str,
    *,
    repo_type: str,
    private: bool = False,
  ) -> None:
    self.repos.append((repo_id, repo_type, private))

  def upload_folder(
    self,
    folder_path: Path,
    repo_id: str,
    *,
    repo_type: str,
    commit_message: str,
  ) -> str:
    self.uploads.append((folder_path, repo_id, repo_type, commit_message))
    return self.upload_revision

  def create_tag(
    self,
    repo_id: str,
    *,
    tag: str,
    revision: str,
    repo_type: str,
  ) -> None:
    self.tags.append((repo_id, tag, revision, repo_type))

  def update_collection_metadata(
    self,
    collection_slug: str,
    *,
    title: str | None = None,
    description: str | None = None,
  ) -> None:
    self.collection_updates.append((collection_slug, title, description))

  def add_collection_item(
    self,
    collection_slug: str,
    *,
    item_id: str,
    item_type: str,
    note: str | None = None,
  ) -> None:
    self.collection_items.append((collection_slug, item_id, item_type, note))

  def get_collection(self, collection_slug: str):
    return self.collections.get(collection_slug)


class HuggingFacePublishClient:
  def create_repo(
    self,
    repo_id: str,
    *,
    repo_type: str,
    private: bool = False,
  ) -> None:
    from huggingface_hub import HfApi

    HfApi().create_repo(
      repo_id=repo_id,
      repo_type=repo_type,
      private=private,
      exist_ok=True,
    )

  def upload_folder(
    self,
    folder_path: Path,
    repo_id: str,
    *,
    repo_type: str,
    commit_message: str,
  ) -> str:
    from huggingface_hub import HfApi

    info = HfApi().upload_folder(
      folder_path=str(folder_path),
      repo_id=repo_id,
      repo_type=repo_type,
      commit_message=commit_message,
    )
    sha = getattr(info, "oid", None) or getattr(info, "commit_sha", None)
    if not sha and hasattr(info, "commit_url"):
      sha = str(info.commit_url).rsplit("/", 1)[-1]
    if not sha:
      raise ValueError(f"upload to {repo_id} did not return a revision sha")
    return str(sha)

  def create_tag(
    self,
    repo_id: str,
    *,
    tag: str,
    revision: str,
    repo_type: str,
  ) -> None:
    from huggingface_hub import HfApi

    HfApi().create_tag(
      repo_id=repo_id,
      tag=tag,
      revision=revision,
      repo_type=repo_type,
      exist_ok=True,
    )

  def update_collection_metadata(
    self,
    collection_slug: str,
    *,
    title: str | None = None,
    description: str | None = None,
  ) -> None:
    from huggingface_hub import HfApi

    HfApi().update_collection_metadata(
      collection_slug=collection_slug,
      title=title,
      description=description,
    )

  def add_collection_item(
    self,
    collection_slug: str,
    *,
    item_id: str,
    item_type: str,
    note: str | None = None,
  ) -> None:
    from huggingface_hub import HfApi

    HfApi().add_collection_item(
      collection_slug=collection_slug,
      item_id=item_id,
      item_type=item_type,
      note=note,
      exists_ok=True,
    )

  def get_collection(self, collection_slug: str):
    from huggingface_hub import HfApi

    return HfApi().get_collection(collection_slug)


_default_client: PublishClient | None = None


def get_default_publish_client() -> PublishClient:
  global _default_client
  if _default_client is None:
    _default_client = HuggingFacePublishClient()
  return _default_client


def set_default_publish_client(client: PublishClient | None) -> None:
  global _default_client
  _default_client = client


def upload_enabled(*, upload_flag: bool) -> bool:
  import os

  if upload_flag:
    return True
  return os.environ.get("HF_PUBLISH", "").strip().lower() in {"1", "true", "yes"}
