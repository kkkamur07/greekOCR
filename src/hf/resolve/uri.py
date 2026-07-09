"""Parse hf:// weights source URIs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HfWeightsUri:
  namespace: str
  hub_repo_slug: str
  registry_tag: str

  @property
  def repo_id(self) -> str:
    return f"{self.namespace}/{self.hub_repo_slug}"


def parse_hf_weights_uri(uri: str) -> HfWeightsUri:
  if not uri.startswith("hf://"):
    raise ValueError(f"unsupported weights source scheme: {uri}")

  rest = uri.removeprefix("hf://")
  if not rest:
    raise ValueError("hf weights source must be hf://<namespace>/<hub-repo-slug>@<registry-tag>")
  if "@" not in rest:
    raise ValueError("hf weights source must include @<registry-tag>")

  repo_part, registry_tag = rest.rsplit("@", 1)
  if not registry_tag:
    raise ValueError("hf weights source must include @<registry-tag>")
  if "/" not in repo_part:
    raise ValueError("hf weights source must be hf://<namespace>/<hub-repo-slug>@<registry-tag>")

  namespace, hub_repo_slug = repo_part.split("/", 1)
  if not namespace or not hub_repo_slug:
    raise ValueError("hf weights source must be hf://<namespace>/<hub-repo-slug>@<registry-tag>")

  return HfWeightsUri(
    namespace=namespace,
    hub_repo_slug=hub_repo_slug,
    registry_tag=registry_tag,
  )
