"""HTTP-level tests for POST /ask (service is replaced by a stub)."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402
from backend.rag.api import get_service  # noqa: E402
from backend.rag.llm import LLMError  # noqa: E402
from backend.rag.retriever import IndexNotReadyError, Retriever  # noqa: E402
from backend.rag.service import AskService  # noqa: E402
from fakes import CHUNKS, FakeCollection, FakeEmbedder, FakeLLM  # noqa: E402


def client_with(service):
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


def real_service(reply="Tiefe nach Wandhöhe [1]."):
    return AskService(Retriever(FakeEmbedder(), FakeCollection(CHUNKS)), FakeLLM(reply), 0.1, 3)


def teardown_function(_):
    app.dependency_overrides.clear()


def test_ask_answered_shape():
    r = client_with(real_service()).post("/ask", json={"question": "Wie tief müssen Abstandsflächen sein?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered" and body["answer"].endswith("[1].")
    s = body["sources"][0]
    assert s["cited"] and s["page_start"] == 12 and s["section_ref"] == "Art. 6" and "PDF-Seite 12" in s["reference"]


def test_ask_no_answer_is_http_200():
    r = client_with(real_service(reply="NO_ANSWER")).post("/ask", json={"question": "Wie tief müssen Abstandsflächen sein?"})
    assert r.status_code == 200 and r.json()["status"] == "no_answer" and r.json()["reason"] == "llm_declined"


def test_validation():
    c = client_with(real_service())
    assert c.post("/ask", json={"question": "ab"}).status_code == 422
    assert c.post("/ask", json={"question": "Gültige Frage?", "k": 0}).status_code == 422
    assert c.post("/ask", json={}).status_code == 422


def test_errors_map_to_http_codes():
    class Boom:
        def __init__(self, exc):
            self.exc = exc

        def ask(self, *a, **k):
            raise self.exc

    assert client_with(Boom(IndexNotReadyError("no index"))).post("/ask", json={"question": "Frage hier?"}).status_code == 503
    assert client_with(Boom(LLMError("upstream"))).post("/ask", json={"question": "Frage hier?"}).status_code == 502


def test_health_still_works():
    assert TestClient(app).get("/health").json() == {"status": "ok"}
