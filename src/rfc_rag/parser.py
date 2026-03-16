from __future__ import annotations

import re

from rfc_rag.models import Section


HEADING_PATTERN = re.compile(
    r"^(?P<number>(?:\d+\.)*\d+\.|Appendix [A-Z]\.)(?:\s{2,}|\s+)(?P<title>.+?)\s*$"
)


def normalize_text(raw_text: str) -> str:
    text = raw_text.lstrip("\ufeff")
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def slugify_section(label: str) -> str:
    lowered = label.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or "section"


def parse_sections(raw_text: str) -> list[Section]:
    text = normalize_text(raw_text)
    lines = text.splitlines()

    sections: list[Section] = []
    current_label = "frontmatter"
    current_lines: list[str] = []

    for line in lines:
        heading_match = HEADING_PATTERN.match(line)
        if heading_match:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append(
                    Section(
                        label=current_label,
                        slug=slugify_section(current_label),
                        content=content,
                    )
                )

            number = heading_match.group("number").rstrip(".")
            title = heading_match.group("title")
            current_label = f"{number} | {title}"
            current_lines = []
            continue

        current_lines.append(line)

    trailing_content = "\n".join(current_lines).strip()
    if trailing_content:
        sections.append(
            Section(
                label=current_label,
                slug=slugify_section(current_label),
                content=trailing_content,
            )
        )

    return sections
