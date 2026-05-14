import re


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    return len(TOKEN_PATTERN.findall(str(text)))


def tokenize_text(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text))


def tokenize_with_spans(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in TOKEN_PATTERN.finditer(str(text))]


def build_chunks(note: str, chunk_size: int = 384, chunk_overlap: int = 64):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    tokens = tokenize_with_spans(note)
    if not tokens:
        return []

    chunks = []
    step = chunk_size - chunk_overlap
    start = 0
    while start < len(tokens):
        if start > 0:
            while start < len(tokens) - 1 and re.fullmatch(r"[^\w\s]+", tokens[start][0]):
                start += 1

        chunk_token_spans = tokens[start:start + chunk_size]
        if not chunk_token_spans:
            continue
        chunk_start = chunk_token_spans[0][1]
        chunk_end = chunk_token_spans[-1][2]
        chunk_text = re.sub(r"_+", "", str(note)[chunk_start:chunk_end]).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if start + chunk_size >= len(tokens):
            break
        start += step

    return [chunk for chunk in chunks if chunk.strip()]