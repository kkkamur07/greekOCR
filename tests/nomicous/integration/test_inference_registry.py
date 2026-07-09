"""Hosted inference registry endpoint for helper sync."""

from fastapi.testclient import TestClient


def test_inference_registry_returns_yaml(client: TestClient):
    response = client.get("/inference/v1/registry")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/yaml")
    assert "models:" in response.text
    assert "greek-calamari-v1:" in response.text
    assert "etag" in response.headers


def test_inference_registry_returns_304_when_etag_matches(client: TestClient):
    first = client.get("/inference/v1/registry")
    etag = first.headers["etag"]
    second = client.get("/inference/v1/registry", headers={"If-None-Match": etag})
    assert second.status_code == 304
    assert second.text == ""


def test_inference_registry_etag_is_stable(client: TestClient):
    first = client.get("/inference/v1/registry")
    second = client.get("/inference/v1/registry")
    assert first.headers["etag"] == second.headers["etag"]
