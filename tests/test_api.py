import pytest
from fastapi.testclient import TestClient

from claimvalue import api
from tests.test_training import trained  # noqa: F401  (session fixture)

CASE = {
    "uf": "SP",
    "comarca": "Campinas",
    "vara": "1ª Vara Cível",
    "judge_id": "JUDGE_unknown",
    "filing_month": 5,
}


@pytest.fixture()
def client(trained, monkeypatch):  # noqa: F811
    model_dir, _ = trained
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("API_KEY", "test-key")
    api._load.cache_clear()
    return TestClient(api.app)


def test_health_is_public(client):
    assert client.get("/health").json()["status"] == "ok"


def test_prediction_requires_key(client):
    assert client.post("/predict/claim-value", json=CASE).status_code == 401
    assert (
        client.post("/predict/claim-value", json=CASE, headers={"X-API-Key": "nope"}).status_code
        == 401
    )


def test_prediction_works_for_unseen_judge_and_missing_features(client):
    r = client.post("/predict/claim-value", json=CASE, headers={"X-API-Key": "test-key"})
    assert r.status_code == 200 and r.json()["predicted_claim_value"] > 0
    r = client.post(
        "/predict/unfavorable-probability", json=CASE, headers={"X-API-Key": "test-key"}
    )
    assert 0 <= r.json()["probability_unfavorable"] <= 1


def test_input_validation(client):
    bad = {**CASE, "filing_month": 13}
    r = client.post("/predict/claim-value", json=bad, headers={"X-API-Key": "test-key"})
    assert r.status_code == 422


def test_server_without_key_fails_closed(trained, monkeypatch):  # noqa: F811
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.chdir(trained[0])  # no .env here
    r = TestClient(api.app).post("/predict/claim-value", json=CASE, headers={"X-API-Key": "x"})
    assert r.status_code == 503
