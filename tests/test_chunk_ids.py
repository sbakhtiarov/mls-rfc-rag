from rfc_rag.chunking import chunk_sections
from rfc_rag.models import Section


def test_chunk_ids_are_unique_within_a_run() -> None:
    sections = [
        Section(label="frontmatter", slug="frontmatter", content="alpha beta gamma"),
        Section(label="1 | Introduction", slug="1-introduction", content="delta " * 50),
    ]

    chunks = chunk_sections(sections, source="rfc9420.txt", strategy="fixed", chunk_size=40)
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))
    assert chunk_ids[0] == "fixed:frontmatter:0"
