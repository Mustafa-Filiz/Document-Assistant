"""Pipeline tests on a synthetic statute PDF (no real documents, no network, no models needed).

Needs `reportlab` and `pypdf` (dev group). Run: uv run pytest
"""
import os
import tempfile

from backend.ingestion import quality
from backend.ingestion.chunking import build_chunks, split_long
from backend.ingestion.extract import (
    extract_document, is_tabular, match_hierarchy, match_section_heading,
)

LONG = " ".join(
    f"Satz {i} regelt, dass bauliche Anlagen den Anforderungen an Standsicherheit entsprechen müssen."
    for i in range(1, 26)
)  # > 1500 chars


def _pages():
    toc = ["Testbauordnung (TestBO)", "Inhaltsübersicht"] + [f"Art. {i} Titel Nummer {i}" for i in range(1, 11)]
    head = lambda n: ["Testbauordnung - Stand 2026", f"Seite {n} von 4"]
    p2 = ["Erster Teil", "Allgemeine Vorschriften",
          "Art. 1 Anwendungsbereich",
          "(1) Dieses Gesetz gilt für bau-", "rechtliche Anlagen und Einrichtungen.",
          "(2) Dieses Gesetz gilt nicht für:", "1. Anlagen des öffentlichen Verkehrs,", "2. Leitungen aller Art.",
          "(+++ Textnachweis ab: 1.1.2025 +++)",
          "Art. 2 Begriffe",
          "(1) Gebäude sind selbstständig benutzbare Anlagen.",
          "Art. 1 Absatz 2 gilt entsprechend. Weitere Regeln folgen später im Text."]
    p3 = ["(2) Gebäudeklassen:", "Klasse Höhe Einheiten", "1 7 2", "2 7 400", "3 7 400", "4 13 400",
          "1) Geändert durch Gesetz vom 1. Januar 2025",
          "Zweiter Teil Bauausführung",
          "Art. 3", "Standsicherheit",
          "(1) " + LONG[:90]]
    p4 = [LONG[90:1200], LONG[1200:], "Art. 4 Brandschutz", "(1) Bauliche Anlagen müssen brandsicher sein.",
          "(2) Weggefallen"]
    pages = [toc, p2, p3, p4]
    return [(h and (head(i + 1) if i else []) or []) + body for i, body in enumerate(pages) for h in [True]]


def _make_pdf(path):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(path, pagesize=A4)
    for lines in _pages():
        y = 800
        for t in lines:
            c.setFont("Helvetica", 9)
            # break very long lines manually so the PDF stays readable
            while len(t) > 100:
                cut = t.rfind(" ", 0, 100)
                c.drawString(40, y, t[:cut]); y -= 12; t = t[cut + 1:]
            c.drawString(40, y, t); y -= 12
        c.showPage()
    c.save()


def _sections():
    with tempfile.TemporaryDirectory() as d:
        pdf = os.path.join(d, "t.pdf")
        _make_pdf(pdf)
        return extract_document(pdf, "testbo")


def test_heading_detection_rejects_in_text_reference():
    assert match_section_heading("Art. 6 Abstandsflächen, Abstände") == ("Art. 6", "Abstandsflächen, Abstände")
    assert match_section_heading("§ 34 Zulässigkeit von Vorhaben") == ("§ 34", "Zulässigkeit von Vorhaben")
    assert match_section_heading("§ 34 Absatz 1 gilt entsprechend") is None
    assert match_hierarchy("Zweiter Teil Besondere Vorschriften")[0] == 1
    assert match_hierarchy("Abschnitt 2 gilt nur für") is None
    assert is_tabular("2 7 400") and not is_tabular("§ 34")


def test_extraction_structure_and_metadata():
    sections, info = _sections()
    assert [s.ref for s in sections] == ["Art. 1", "Art. 2", "Art. 3", "Art. 4"]  # ToC page ignored
    by = {s.ref: s for s in sections}
    assert by["Art. 1"].hierarchy == ["Erster Teil Allgemeine Vorschriften"]
    assert by["Art. 3"].title == "Standsicherheit"  # ref-only heading + title on next line
    assert by["Art. 3"].hierarchy == ["Zweiter Teil Bauausführung"]
    assert info["amendment_notes_dropped"] == 1
    text = " ".join(a.text for s in sections for a in s.atoms)
    assert "baurechtliche" in text                      # de-hyphenated
    assert "Seite" not in text and "Stand 2026" not in text  # running header/footer removed
    # the wrapped in-text reference stayed inside Art. 2 instead of opening a new section
    assert "Art. 1 Absatz 2 gilt entsprechend" in " ".join(a.text for a in by["Art. 2"].atoms)
    # page numbers
    assert by["Art. 1"].page_start == 2 and by["Art. 4"].page_start == 4


def test_clause_chunks():
    sections, _ = _sections()
    chunks = build_chunks(sections, "testbo", "TestBO", "clause", max_chars=1500)
    assert all(c.section_ref for c in chunks)
    assert all(c.char_len <= 1500 for c in chunks)
    assert not any("Weggefallen" in c.text for c in chunks)            # stub dropped
    long_parts = [c for c in chunks if c.section_ref == "Art. 3"]
    assert len(long_parts) >= 2 and all(c.split_within_clause for c in long_parts)
    assert all(c.embed_text.startswith("TestBO | ") for c in chunks)
    plain = build_chunks(sections, "testbo", "TestBO", "clause_plain")
    assert [c.text for c in plain] == [c.text for c in chunks] and plain[0].embed_text == plain[0].text
    art2 = [c for c in chunks if c.section_ref == "Art. 2"]
    assert any("table_like" in quality.flags(c) for c in art2)
    assert any("footnote_or_amendment_note" in quality.flags(c) for c in art2)
    assert not quality.ordering_anomalies(chunks)


def test_fixed_chunks_track_sections_and_pages():
    sections, _ = _sections()
    chunks = build_chunks(sections, "testbo", "TestBO", "fixed")
    assert len(chunks) >= 3
    assert chunks[0].section_ref == "Art. 1" and chunks[0].page_start == 2
    assert any("|" in c.refs_covered for c in chunks)            # windows that straddle two sections
    assert any("crosses_section" in quality.flags(c) for c in chunks)


def test_split_long_keeps_abbreviations_together():
    pieces = split_long("Die Abstandsfläche nach Art. 6 Abs. 1 Satz 1 ist einzuhalten. " * 60, 500)
    assert all(len(p) <= 500 for p in pieces)
    assert not any(p.rstrip().endswith("Abs.") for p in pieces)
