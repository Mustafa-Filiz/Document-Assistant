"""RAG pipeline tests (no model, no network, no LLM key). Run: uv run pytest"""
import pytest

from backend.config import Settings
from backend.rag import llm as llm_mod
from backend.rag.calibrate import suggest_threshold
from backend.rag.cli import check_flags
from backend.rag.lang import detect_language
from backend.rag.llm import AnthropicClient, LLMError, OpenAICompatClient, build_llm_client
from backend.rag.prompts import SYSTEM_PROMPT, build_user_prompt, reference
from backend.rag.retriever import Hit, Retriever
from backend.rag.service import AskService, extract_citations, is_no_answer
from fakes import CHUNKS, FakeCollection, FakeEmbedder, FakeLLM

Q_ABSTAND = "Wie tief müssen Abstandsflächen sein?"


def make_service(reply="Die Tiefe richtet sich nach der Wandhöhe [1].", threshold=0.1, llm=True):
    retr = Retriever(FakeEmbedder(), FakeCollection(CHUNKS))
    fake = FakeLLM(reply) if llm else None
    return AskService(retr, fake, min_similarity=threshold, default_k=3), fake


# ---------------------------------------------------------------- retrieval
def test_search_ranks_best_chunk_first_and_scores_are_cosine():
    hits = Retriever(FakeEmbedder(), FakeCollection(CHUNKS)).search(Q_ABSTAND, k=3)
    assert hits[0].meta["section_ref"] == "Art. 6"
    assert hits[0].score > hits[1].score >= hits[2].score
    assert 0 < hits[0].score <= 1


def test_search_doc_filter_and_k_larger_than_index():
    r = Retriever(FakeEmbedder(), FakeCollection(CHUNKS))
    assert {h.meta["doc_id"] for h in r.search(Q_ABSTAND, k=10, doc_id="geg")} == {"geg"}
    assert len(r.search(Q_ABSTAND, k=10)) == 3
    assert r.search(Q_ABSTAND, k=3, doc_id="nope") == []
    with pytest.raises(ValueError):
        r.search(Q_ABSTAND, k=0)


# ---------------------------------------------------------------- the three "I don't know" paths
def test_below_threshold_never_calls_llm():
    svc, fake = make_service(threshold=0.5)
    res = svc.ask("Wie backe ich einen Apfelkuchen?")
    assert (res.status, res.reason) == ("no_answer", "below_threshold")
    assert fake.calls == []
    assert res.answer.startswith("Ich weiß es nicht")
    assert res.sources and not any(s.in_context for s in res.sources)  # shown for debugging, flagged


def test_unknown_doc_filter_gives_no_results():
    res = make_service()[0].ask(Q_ABSTAND, doc_id="nope")
    assert (res.status, res.reason, res.top_score) == ("no_answer", "no_results", None)


def test_llm_declines():
    res = make_service(reply="NO_ANSWER")[0].ask(Q_ABSTAND)
    assert (res.status, res.reason) == ("no_answer", "llm_declined")
    assert not any(s.cited for s in res.sources)


def test_uncited_answer_is_discarded():
    res = make_service(reply="Die Tiefe beträgt 3 Meter.")[0].ask(Q_ABSTAND)
    assert (res.status, res.reason) == ("no_answer", "uncited_answer")
    assert "3 Meter" not in res.answer


def test_citation_to_nonexistent_source_does_not_count():
    res = make_service(reply="Das steht in [9].")[0].ask(Q_ABSTAND)
    assert res.reason == "uncited_answer"


def test_no_answer_message_follows_question_language():
    svc, _ = make_service(threshold=0.9)
    assert svc.ask("Elma pastası nasıl yapılır?").answer.startswith("Bilmiyorum")
    assert svc.ask("What is the best apple pie recipe?").answer.startswith("I don't know")


# ---------------------------------------------------------------- answered path + references
def test_answered_returns_page_and_section_references():
    svc, fake = make_service()
    res = svc.ask(Q_ABSTAND)
    assert res.status == "answered" and res.reason is None and res.model == "fake-llm"
    top = res.sources[0]
    assert top.cited and top.in_context
    assert top.reference == "Bayerische Bauordnung (BayBO), Art. 6 Abstandsflächen, Abstände, Absatz (1), PDF-Seite 12"
    assert (top.page_start, top.page_end) == (12, 12)
    system, user = fake.calls[0]
    assert system == SYSTEM_PROMPT
    assert "<quellen>" in user and "[1] Bayerische Bauordnung (BayBO) | Art. 6" in user
    assert user.endswith(f"Frage: {Q_ABSTAND}")


def test_only_in_context_sources_go_to_the_llm():
    svc, fake = make_service(threshold=0.15)
    res = svc.ask(Q_ABSTAND)
    given = [s for s in res.sources if s.in_context]
    assert 1 <= len(given) < len(res.sources)
    user = fake.calls[0][1]
    assert f"[{len(given)}] " in user and f"[{len(given) + 1}] " not in user
    assert "Energieausweis" not in fake.calls[0][1]


def test_retrieval_only_modes():
    res = make_service(llm=False)[0].ask(Q_ABSTAND)
    assert (res.status, res.reason, res.answer) == ("retrieval_only", "no_llm_configured", "")
    svc, fake = make_service()
    res = svc.ask(Q_ABSTAND, generate=False)
    assert (res.status, res.reason) == ("retrieval_only", "generation_disabled") and fake.calls == []


def test_empty_question_rejected():
    with pytest.raises(ValueError):
        make_service()[0].ask("   ")


# ---------------------------------------------------------------- helpers
def test_extract_citations_formats():
    assert extract_citations("a [1] b [2][3] c [4, 5]; d [6;7] e [0] f [8]", max_n=7) == {1, 2, 3, 4, 5, 6, 7}
    assert extract_citations("no cites, Absatz (1)", 5) == set()


def test_is_no_answer_variants():
    assert all(is_no_answer(x) for x in ["NO_ANSWER", " no_answer.", "`NO_ANSWER`", "NO_ANSWER\n"])
    assert not is_no_answer("Es gibt keine NO_ANSWER hier")


def test_detect_language():
    assert detect_language("Bir binanın sınırdan mesafesi ne kadar olmalı?") == "tr"
    assert detect_language("Elma pastası nasıl yapılır?") == "tr"
    assert detect_language("When does a building need an energy certificate?") == "en"
    assert detect_language("Wann darf im Außenbereich gebaut werden?") == "de"


def test_reference_without_section_or_absatz():
    m = {"doc_name": "GEG", "section_ref": "", "section_title": "", "absatz": "-", "page_start": 1, "page_end": 7}
    assert reference(m) == "GEG, PDF-Seiten 1–7"


def test_user_prompt_numbering():
    hits = [Hit(0.9, "Text A", CHUNKS[0][1]), Hit(0.8, "Text B", CHUNKS[1][1])]
    p = build_user_prompt("Frage?", hits)
    assert "[1] " in p and "[2] " in p and "[3] " not in p and "Text B" in p


# ---------------------------------------------------------------- LLM clients (injected transport)
def test_anthropic_client_request_and_parsing():
    seen = {}

    def post(url, headers, payload, timeout):
        seen.update(url=url, headers=headers, payload=payload)
        return {"content": [{"type": "text", "text": "Antwort [1]"}, {"type": "tool_use"}]}

    c = AnthropicClient("sk-test", model="m1", max_tokens=50, post=post)
    assert c.complete("sys", "usr") == "Antwort [1]"
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "sk-test" and "anthropic-version" in seen["headers"]
    assert seen["payload"] == {"model": "m1", "max_tokens": 50, "temperature": 0, "system": "sys",
                               "messages": [{"role": "user", "content": "usr"}]}
    with pytest.raises(LLMError):
        AnthropicClient("k", post=lambda *a: {"content": []}).complete("s", "u")


def test_openai_compat_client():
    def post(url, headers, payload, timeout):
        assert url == "http://localhost:8080/v1/chat/completions" and headers["Authorization"] == "Bearer k"
        assert payload["messages"][0] == {"role": "system", "content": "sys"}
        return {"choices": [{"message": {"content": " ok [1] "}}]}

    c = OpenAICompatClient("k", "m", base_url="http://localhost:8080/v1/", post=post)
    assert c.complete("sys", "usr") == "ok [1]"
    with pytest.raises(LLMError):
        OpenAICompatClient("k", "m", post=lambda *a: {"oops": 1}).complete("s", "u")


def test_build_llm_client_selection(monkeypatch=None):
    cfg = lambda **kw: Settings(**{"llm_api_key": "", "llm_provider": "anthropic", "llm_model": "", **kw})
    assert build_llm_client(cfg()) is None
    assert build_llm_client(cfg(llm_api_key="k")).model == llm_mod.DEFAULT_ANTHROPIC_MODEL
    assert build_llm_client(cfg(llm_api_key="k", llm_provider="openai", llm_model="x")).model == "x"
    with pytest.raises(ValueError):
        build_llm_client(cfg(llm_api_key="k", llm_provider="openai"))
    with pytest.raises(ValueError):
        build_llm_client(cfg(llm_api_key="k", llm_provider="nope"))


# ---------------------------------------------------------------- calibration + trial flags
def test_suggest_threshold():
    t, note = suggest_threshold([0.86, 0.88, 0.90], [0.70, 0.74, 0.76])
    assert 0.76 < t < 0.86 and note == "clean separation"
    t, note = suggest_threshold([0.80, 0.90, 0.92], [0.78, 0.82])
    assert "OVERLAP" in note
    with pytest.raises(ValueError):
        suggest_threshold([], [0.5])


def test_trial_flags():
    svc, _ = make_service()
    res = svc.ask(Q_ABSTAND)
    assert check_flags({"expect": "answer", "expected_doc": "baybo", "expected_ref": "Art. 6"}, res) == []
    assert check_flags({"expect": "decline_oos"}, res)[0].startswith("ANSWERED-BUT-SHOULD-DECLINE")
    assert "RETRIEVAL-MISS" in check_flags({"expect": "answer", "expected_doc": "geg", "expected_ref": "§ 71"}, res)[0]
    declined = make_service(reply="NO_ANSWER")[0].ask(Q_ABSTAND)
    assert check_flags({"expect": "answer"}, declined)[0].startswith("OVER-REFUSAL")
    assert check_flags({"expect": "decline"}, declined) == []
