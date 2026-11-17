"""In-memory stand-ins so the RAG pipeline is testable without Chroma, torch or a network."""
import math
import re
import zlib

DIM = 4096


def _vec(text):
    v = [0.0] * DIM
    for w in re.findall(r"\w+", text.lower()):
        v[zlib.crc32(w.encode()) % DIM] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class FakeEmbedder:
    def embed_queries(self, texts):
        return [_vec(t) for t in texts]


class FakeCollection:
    """Implements the slice of the Chroma API that store.query() uses (cosine distance)."""

    def __init__(self, chunks):
        self.chunks = chunks  # list of (text, meta)
        self.vecs = [_vec(t) for t, _ in chunks]

    def count(self):
        return len(self.chunks)

    def query(self, query_embeddings, n_results, where=None, include=None):
        q = query_embeddings[0]
        rows = []
        for (text, meta), v in zip(self.chunks, self.vecs):
            if where and any(meta.get(k) != val for k, val in where.items()):
                continue
            rows.append((1.0 - sum(a * b for a, b in zip(q, v)), text, meta))
        rows.sort(key=lambda r: r[0])
        rows = rows[:n_results]
        return {
            "documents": [[r[1] for r in rows]],
            "metadatas": [[r[2] for r in rows]],
            "distances": [[r[0] for r in rows]],
        }


class FakeLLM:
    model = "fake-llm"

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.reply


def meta(doc_id, doc_name, ref, title, p0, p1=None, absatz="(1)"):
    return {"doc_id": doc_id, "doc_name": doc_name, "section_ref": ref, "section_title": title,
            "hierarchy": "", "absatz": absatz, "page_start": p0, "page_end": p1 or p0}


CHUNKS = [
    ("Abstandsflächen Tiefe Wand Höhe Meter Gebäude einhalten",
     meta("baybo", "Bayerische Bauordnung (BayBO)", "Art. 6", "Abstandsflächen, Abstände", 12)),
    ("Energieausweis ausgestellt Gebäude Verkauf Vermietung",
     meta("geg", "Gebäudeenergiegesetz (GEG)", "§ 80", "Ausstellung von Energieausweisen", 40, 41)),
    ("Außenbereich Vorhaben zulässig privilegiert öffentliche Belange",
     meta("baugb", "Baugesetzbuch (BauGB)", "§ 35", "Bauen im Außenbereich", 20, 21, absatz="(1)-(2)")),
]
