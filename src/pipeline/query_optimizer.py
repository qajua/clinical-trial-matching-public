import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm.vllm_client import VLLMClient


FIXED_QUERY_OPTIMIZATIONS_PATH = Path("outputs/query_optimizations.json")

QOPT_SYSTEM_PROMPT = """\
You are a clinical NLP assistant specializing in medical eligibility question analysis.

Given a medical eligibility question entered by a user, output a JSON object with exactly these three fields:
- "rewritten_query": a single clear sentence optimized for document retrieval (focus on finding matching evidence in clinical notes)
- "sub_queries": a list of 2-4 focused sub-questions that together cover all aspects needed to answer the original question
- "retrieval_query": space-separated key medical terms for BM25/vector search (abbreviations, synonyms, concept names)

Output ONLY valid JSON. No explanation, no markdown fences.
"""


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def build_medical_question(question: str) -> str:
    return normalize_text(question)


def _load_fixed_query_optimizations() -> dict:
    if not FIXED_QUERY_OPTIMIZATIONS_PATH.exists():
        return {}

    try:
        with FIXED_QUERY_OPTIMIZATIONS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[qopt] Failed to load fixed query optimizations, ignoring file: {e}")
        return {}


def _rule_based_optimize(question: str) -> dict:
    """LLM 失败时的 fallback。"""
    q = normalize_text(question).lower()

    base_terms = re.findall(r"[a-zA-Z0-9_]+", q)
    stopwords = {
        "does", "the", "note", "describe", "patient", "as", "having", "have",
        "a", "an", "or", "of", "and", "to", "is", "was", "with", "on", "ever",
        "what", "highest", "answer", "mentioned", "if", "no", "yes", "there", "any",
        "please", "can", "could", "would", "should", "for", "from", "into"
    }
    base_terms = [t for t in base_terms if t not in stopwords and len(t) > 2]

    merged = []
    seen = set()
    for item in base_terms + ["clinical note evidence"]:
        key = item.lower()
        if key not in seen:
            merged.append(item)
            seen.add(key)

    retrieval_query = " ".join(merged)
    rewritten_query = normalize_text(question)
    sub_queries = [
        rewritten_query,
        f"What evidence in the note helps answer: {rewritten_query}",
    ]

    return {
        "rewritten_query": rewritten_query,
        "sub_queries": sub_queries,
        "retrieval_query": retrieval_query,
    }


def optimize_query(
    question: str,
    llm: "VLLMClient | None" = None,
) -> dict:
    """
    Query optimization: rewrite + decompose.

    Returns:
        {
            "rewritten_query": str,
            "sub_queries": list[str],
            "retrieval_query": str,
        }

    Falls back to rule-based implementation if llm is None or LLM call fails.
    """
    fixed_result = _load_fixed_query_optimizations().get(str(question).strip())
    if fixed_result is not None:
        return fixed_result

    if llm is None:
        return _rule_based_optimize(question)

    user_prompt = f"Question: {normalize_text(question)}"

    try:
        raw = llm.generate(
            system_prompt=QOPT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=512,
        ).strip()

        # LLM sometimes wraps output in ```json ... ```
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)

        parsed = json.loads(raw)

        rewritten_query = str(parsed.get("rewritten_query", question)).strip()
        sub_queries = parsed.get("sub_queries", [])
        if not isinstance(sub_queries, list) or not sub_queries:
            sub_queries = [rewritten_query]
        sub_queries = [str(q).strip() for q in sub_queries if str(q).strip()]

        retrieval_query = str(parsed.get("retrieval_query", rewritten_query)).strip()

        return {
            "rewritten_query": rewritten_query,
            "sub_queries": sub_queries,
            "retrieval_query": retrieval_query,
        }

    except Exception as e:
        print(f"[qopt] LLM optimize_query failed, using rule-based fallback: {e}")
        return _rule_based_optimize(question)