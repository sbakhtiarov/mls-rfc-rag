from pathlib import Path

from rfc_rag.parser import parse_sections


RFC_PATH = Path(__file__).resolve().parents[1] / "rfc9420.txt"


def test_parse_sections_finds_expected_headings() -> None:
    sections = parse_sections(RFC_PATH.read_text(encoding="utf-8-sig"))
    labels = {section.label for section in sections}

    assert "frontmatter" in labels
    assert "1 | Introduction" in labels
    assert "12.4.3.2 | Joining via External Commits" in labels
    assert "17.1 | MLS Cipher Suites" in labels
    assert "Appendix A | Protocol Origins of Example Trees" in labels


def test_frontmatter_contains_preface_content() -> None:
    sections = parse_sections(RFC_PATH.read_text(encoding="utf-8-sig"))
    frontmatter = sections[0]

    assert frontmatter.label == "frontmatter"
    assert "The Messaging Layer Security (MLS) Protocol" in frontmatter.content
