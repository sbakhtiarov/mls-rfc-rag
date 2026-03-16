from pathlib import Path

from rfc_rag.chunking import chunk_sections
from rfc_rag.parser import parse_sections


RFC_PATH = Path(__file__).resolve().parents[1] / "rfc9420.txt"


def test_fixed_chunking_keeps_chunks_within_size_and_section() -> None:
    sections = parse_sections(RFC_PATH.read_text(encoding="utf-8-sig"))
    chunks = chunk_sections(sections, source="rfc9420.txt", strategy="fixed", chunk_size=300)

    assert chunks
    assert all(chunk.char_count <= 300 for chunk in chunks)
    assert all(chunk.section for chunk in chunks)
    assert all(chunk.source == "rfc9420.txt" for chunk in chunks)
    assert all(chunk.chunk_id.startswith("fixed:") for chunk in chunks)

    intro_chunks = [chunk for chunk in chunks if chunk.section == "1 | Introduction"]
    assert intro_chunks
    assert all(chunk.section == "1 | Introduction" for chunk in intro_chunks)


def test_section_chunking_preserves_short_sections_and_splits_large_ones() -> None:
    sections = parse_sections(RFC_PATH.read_text(encoding="utf-8-sig"))
    chunks = chunk_sections(sections, source="rfc9420.txt", strategy="section", chunk_size=2500)

    intro_chunks = [chunk for chunk in chunks if chunk.section == "1 | Introduction"]
    cipher_chunks = [chunk for chunk in chunks if chunk.section == "17.1 | MLS Cipher Suites"]

    assert len(intro_chunks) == 1
    assert len(cipher_chunks) > 1
    assert all(chunk.chunk_id.startswith("section:") for chunk in chunks)
