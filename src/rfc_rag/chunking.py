from __future__ import annotations

from rfc_rag.models import Chunk, Section


def split_text(text: str, chunk_size: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    stripped = text.strip()
    if not stripped:
        return []

    parts: list[str] = []
    remaining = stripped

    while len(remaining) > chunk_size:
        split_at = remaining.rfind(" ", 0, chunk_size + 1)
        if split_at <= 0:
            split_at = chunk_size

        piece = remaining[:split_at].strip()
        if piece:
            parts.append(piece)

        remaining = remaining[split_at:].lstrip()

    if remaining:
        parts.append(remaining)

    return parts


def chunk_sections(
    sections: list[Section],
    source: str,
    strategy: str,
    chunk_size: int,
) -> list[Chunk]:
    if strategy not in {"fixed", "section"}:
        raise ValueError(f"Unsupported strategy: {strategy}")

    chunks: list[Chunk] = []
    for section in sections:
        if strategy == "fixed":
            section_parts = split_text(section.content, chunk_size)
        else:
            if len(section.content) <= chunk_size:
                section_parts = [section.content]
            else:
                section_parts = split_text(section.content, chunk_size)

        for index, part in enumerate(section_parts):
            chunks.append(
                Chunk(
                    chunk_id=f"{strategy}:{section.slug}:{index}",
                    source=source,
                    section=section.label,
                    chunk_index=index,
                    content=part,
                    char_count=len(part),
                )
            )

    return chunks
