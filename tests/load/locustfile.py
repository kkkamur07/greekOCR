"""Read-heavy Locust scenarios for the hosted platform API.

The default scenarios do not upload files, enqueue jobs, or mutate documents.
Use a pre-created access token for authenticated traffic so the auth rate limit
does not become the bottleneck for the API workload.
"""

import os
from logging import getLogger

from locust import HttpUser, between, task

logger = getLogger(__name__)


def _env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


class PlatformApiUser(HttpUser):
    """Exercise platform API reads and their database-backed dependencies."""

    wait_time = between(1, 3)
    host = _env("LOCUST_HOST") or "https://api.nomicous.com"

    def on_start(self) -> None:
        self.access_token = _env("LOCUST_ACCESS_TOKEN")
        self.project_id = _env("LOCUST_PROJECT_ID")
        self.document_id = _env("LOCUST_DOCUMENT_ID")
        self.part_id = _env("LOCUST_PART_ID")
        self.job_id = _env("LOCUST_JOB_ID")

        if self.access_token:
            return

        email = _env("LOCUST_EMAIL")
        password = _env("LOCUST_PASSWORD")
        if not email or not password:
            return

        with self.client.post(
            "/auth/login",
            json={"email": email, "password": password},
            name="POST /auth/login",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"login returned HTTP {response.status_code}")
                logger.warning(
                    "Unable to authenticate Locust user: HTTP %s",
                    response.status_code,
                )
                return
            self.access_token = response.json().get("access_token")
            if not self.access_token:
                response.failure("login response did not contain access_token")

    @property
    def auth_headers(self) -> dict[str, str]:
        if not self.access_token:
            return {}
        token = self.access_token.removeprefix("Bearer ").strip()
        return {"Authorization": f"Bearer {token}"}

    @task(5)
    def health(self) -> None:
        self.client.get("/health", name="GET /health")

    @task(2)
    def root(self) -> None:
        self.client.get("/", name="GET /")

    @task(2)
    def inference_registry(self) -> None:
        self.client.get("/inference/v1/registry", name="GET /inference/v1/registry")

    @task(4)
    def current_user(self) -> None:
        if not self.access_token:
            return
        self.client.get("/me", headers=self.auth_headers, name="GET /me")

    @task(3)
    def list_projects(self) -> None:
        if not self.access_token:
            return
        self.client.get("/projects", headers=self.auth_headers, name="GET /projects")

    @task(3)
    def project_reads(self) -> None:
        if not self.access_token or not self.project_id:
            return
        project_id = self.project_id
        headers = self.auth_headers
        self.client.get(
            f"/projects/{project_id}",
            headers=headers,
            name="GET /projects/{project_id}",
        )
        self.client.get(
            f"/projects/{project_id}/jobs?limit=8",
            headers=headers,
            name="GET /projects/{project_id}/jobs",
        )

    @task(4)
    def list_documents(self) -> None:
        if not self.access_token or not self.project_id:
            return
        self.client.get(
            f"/projects/{self.project_id}/documents?limit=50",
            headers=self.auth_headers,
            name="GET /projects/{project_id}/documents",
        )

    @task(4)
    def document_reads(self) -> None:
        if not self.access_token or not self.project_id or not self.document_id:
            return
        self.client.get(
            f"/projects/{self.project_id}/documents/{self.document_id}",
            headers=self.auth_headers,
            name="GET /projects/{project_id}/documents/{document_id}",
        )
        self.client.get(
            f"/projects/{self.project_id}/documents/{self.document_id}/transcriptions",
            headers=self.auth_headers,
            name="GET /projects/{project_id}/documents/{document_id}/transcriptions",
        )

    @task(4)
    def page_editor_reads(self) -> None:
        if (
            not self.access_token
            or not self.project_id
            or not self.document_id
            or not self.part_id
        ):
            return
        prefix = (
            f"/projects/{self.project_id}/documents/{self.document_id}/parts/{self.part_id}"
        )
        headers = self.auth_headers
        self.client.get(
            f"{prefix}/layout",
            headers=headers,
            name="GET /projects/{project_id}/documents/{document_id}/parts/{part_id}/layout",
        )
        self.client.get(
            f"{prefix}/lines",
            headers=headers,
            name="GET /projects/{project_id}/documents/{document_id}/parts/{part_id}/lines",
        )
        self.client.get(
            f"{prefix}/pairing",
            headers=headers,
            name="GET /projects/{project_id}/documents/{document_id}/parts/{part_id}/pairing",
        )

    @task(3)
    def list_inference_models(self) -> None:
        if not self.access_token:
            return
        self.client.get(
            "/inference/models",
            headers=self.auth_headers,
            name="GET /inference/models",
        )

    @task(1)
    def model_binding_resolution(self) -> None:
        if (
            not self.access_token
            or not self.project_id
            or not self.document_id
            or not self.part_id
        ):
            return
        path = (
            f"/projects/{self.project_id}/documents/{self.document_id}/parts/"
            f"{self.part_id}/model-bindings/resolve?task=transcribe"
        )
        self.client.get(
            path,
            headers=self.auth_headers,
            name="GET .../model-bindings/resolve",
        )

    @task(3)
    def authenticated_media(self) -> None:
        if not self.access_token or not self.part_id:
            return
        self.client.get(
            f"/media/parts/{self.part_id}?w=1200",
            headers=self.auth_headers,
            name="GET /media/parts/{part_id}",
        )

    @task(2)
    def job_status(self) -> None:
        if not self.access_token or not self.job_id:
            return
        self.client.get(
            f"/jobs/{self.job_id}",
            headers=self.auth_headers,
            name="GET /jobs/{job_id}",
        )

    @task(3)
    def public_document_reads(self) -> None:
        if not self.project_id or not self.document_id:
            return
        prefix = f"/public/projects/{self.project_id}/documents/{self.document_id}"
        self.client.get(
            prefix,
            name="GET /public/projects/{project_id}/documents/{document_id}",
        )
        self.client.get(
            f"{prefix}/layout",
            name="GET /public/projects/{project_id}/documents/{document_id}/layout",
        )

    @task(1)
    def public_transcriptions(self) -> None:
        if not self.project_id or not self.document_id:
            return
        self.client.get(
            f"/public/projects/{self.project_id}/documents/{self.document_id}/transcriptions",
            name="GET /public/projects/{project_id}/documents/{document_id}/transcriptions",
        )

    @task(3)
    def public_media(self) -> None:
        if not self.part_id:
            return
        self.client.get(
            f"/public/media/parts/{self.part_id}?w=1200",
            name="GET /public/media/parts/{part_id}",
        )

    @task(1)
    def public_exports(self) -> None:
        if not self.project_id or not self.document_id or not self.part_id:
            return
        prefix = (
            f"/public/projects/{self.project_id}/documents/{self.document_id}/parts/"
            f"{self.part_id}"
        )
        self.client.get(
            f"{prefix}/page-xml",
            name="GET /public/.../page-xml",
        )
        self.client.get(
            f"{prefix}/transcription-pdf",
            name="GET /public/.../transcription-pdf",
        )
