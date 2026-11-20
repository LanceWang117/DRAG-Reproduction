"""Unified answer extractor E(.), applied identically to every method.

The baseline scores Naive RAG after ``single_agent_pred_parse`` but scores DRAG's raw
moderator text (finding B12). E(.) gives every arm the same extraction policy:

1. Text after the **last** "The answer is:" or "Final answer:" (case-insensitive),
   first line only.
2. If that marker has nothing after it, the last non-empty line before the marker.
3. Without a marker, the first non-empty line.
4. Strip surrounding whitespace and markdown emphasis, then trailing punctuation.
5. With ``yes_no=True`` (StrategyQA), map a leading yes/no word to "Yes" / "No".
"""
import re

_MARKER = re.compile(r"(?:the answer is|final answer)\s*:", re.IGNORECASE)
_TRAILING_PUNCTUATION = ".,;:!"
# Map only a bare yes/no or one followed by punctuation: "Yes, because ..." -> "Yes",
# but "No one can say" stays as it is.
_YES_NO = re.compile(r"^\W*(yes|no)(?:\W*$|\s*[,.;:!])", re.IGNORECASE)


def _non_empty_lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def _candidate(text):
    markers = list(_MARKER.finditer(text))
    if not markers:
        lines = _non_empty_lines(text)
        return lines[0] if lines else ""
    last = markers[-1]
    after = _non_empty_lines(text[last.end():])
    if after:
        return after[0]
    before = _non_empty_lines(_MARKER.sub("", text[:last.start()]))
    return before[-1] if before else ""


def _clean(candidate):
    # Punctuation can sit outside emphasis ("**Paris**.") or inside it ("**Paris.**").
    cleaned = candidate.strip().rstrip(_TRAILING_PUNCTUATION).strip()
    cleaned = cleaned.strip("*_").strip()
    return cleaned.rstrip(_TRAILING_PUNCTUATION).strip()


def _to_yes_no(answer):
    match = _YES_NO.match(answer)
    return match.group(1).capitalize() if match else answer


def extract_answer(text, yes_no=False):
    """Return the final answer span from a model output (see the module docstring)."""
    if not text:
        return ""
    answer = _clean(_candidate(text))
    return _to_yes_no(answer) if yes_no else answer
