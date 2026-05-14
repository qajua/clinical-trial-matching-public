import pandas as pd
import requests

from src.config import load_config
from src.llm.vllm_client import VLLMClient
from src.pipeline.answer_generation import AnswerGenerator
from src.pipeline.evaluator import evaluate_predictions
from src.pipeline.langextract import _cache_key, _load_disk_cache
from src.pipeline.query_optimizer import optimize_query
from src.utils.io import save_csv


def _clean_optional_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _parse_top_chunks(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return []
    return [part.strip() for part in text.split(" ||| ") if part.strip()]


def _normalize_merge_keys(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    normalized = df.copy()
    for col in key_cols:
        normalized[col] = normalized[col].map(lambda x: str(x).strip())
    return normalized


def _parse_sub_queries(value) -> list[str]:
    return [
        part.strip()
        for part in str(value or "").split("|")
        if part.strip() and part.strip().lower() != "nan"
    ]


def _row_sub_queries(row) -> list[str]:
    sub_queries = _parse_sub_queries(row.get("sub_queries", ""))
    if sub_queries:
        return sub_queries

    question = str(row.get("question", "")).strip()
    if not question:
        return []
    qopt = optimize_query(question, llm=None)
    sub_queries = qopt.get("sub_queries", [])
    if not isinstance(sub_queries, list):
        return []
    return [str(item).strip() for item in sub_queries if str(item).strip()]


_QUERY_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "without",
    "does", "do", "did", "is", "are", "was", "were", "be", "been", "being", "having",
    "have", "has", "had", "patient", "note", "describe", "mentioned", "answer", "if",
    "no", "yes", "ever", "during", "within", "last", "this", "that", "what", "highest",
    "lowest", "value", "score", "count", "level", "fraction", "lab", "available",
}


def _normalize_text(text: str) -> str:
    import re
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())


def _query_focus_terms(question: str, sub_queries: list[str]) -> list[str]:
    terms = []
    for phrase in sub_queries:
        normalized = _normalize_text(phrase)
        if len(normalized) >= 4:
            terms.append(normalized)
    for token in _normalize_text(question).split():
        if len(token) >= 3 and token not in _QUERY_STOPWORDS:
            terms.append(token)
    return sorted(set(terms), key=len, reverse=True)


def _filter_primekg_evidence(question: str, sub_queries: list[str], top_chunks: list[str], evidence: str) -> str:
    cleaned = _clean_optional_text(evidence)
    if not cleaned:
        return ""

    evidence_text = _normalize_text(cleaned)
    chunk_text = _normalize_text(" ".join(top_chunks))
    focus_terms = _query_focus_terms(question, sub_queries)
    if not focus_terms:
        return ""

    matched_terms = [term for term in focus_terms if term in evidence_text]
    if not matched_terms:
        return ""

    if any(term not in chunk_text for term in matched_terms):
        return cleaned
    return ""


def _numeric_target_patterns(question: str) -> list[str]:
    q = str(question).lower()
    if "glucose" in q:
        return [r"glucose", r"\bglu\b"]
    if "hemoglobin" in q or "hgb" in q:
        return [r"hgb", r"hemoglobin", r"\bhb\b"]
    if "platelet" in q or "plt" in q:
        return [r"plt", r"platelet"]
    if "creatinine" in q or "creat" in q:
        return [r"creat", r"creatinine"]
    if "aspartate aminotransferase" in q or "ast" in q:
        return [r"ast", r"sgot", r"aspartate"]
    if "bilirubin" in q or "totbili" in q or "bili" in q:
        return [r"totbili", r"bili", r"bilirubin"]
    if "left ventricular ejection" in q or "ejection fraction" in q or "lvef" in q:
        return [r"lvef", r"\bef\b", r"ejection fraction"]
    if "chads2" in q:
        return [r"chads2", r"\bchads\b"]
    if "cha2ds2" in q:
        return [r"cha2ds2", r"cha2ds2[- ]?vasc", r"\bvasc\b"]
    return []


def _is_lowest_question(question: str) -> bool:
    return "lowest" in str(question).lower() or "minimum" in str(question).lower()


def _valid_numeric_value(question: str, value: float) -> bool:
    q = str(question).lower()
    if "glucose" in q:
        return 20 <= value <= 1000
    if "hemoglobin" in q or "hgb" in q:
        return 1 <= value <= 25
    if "platelet" in q or "plt" in q:
        return 1 <= value <= 2000
    if "creatinine" in q or "creat" in q:
        return 0.1 <= value <= 30
    if "ast" in q or "aspartate aminotransferase" in q:
        return 1 <= value <= 10000
    if "bilirubin" in q or "totbili" in q or "bili" in q:
        return 0 <= value <= 100
    if "ejection fraction" in q or "lvef" in q or "left ventricular ejection" in q:
        return 1 <= value <= 100
    if "chads2" in q or "cha2ds2" in q:
        return 0 <= value <= 20
    return True


def _format_numeric_value(value: float) -> str:
    text = f"{value:.10g}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _is_spurious_numeric_match(question: str, text: str, match) -> bool:
    import re
    q = str(question).lower()
    suffix = str(text)[match.end(1):match.end(1) + 8].lower()
    if suffix.startswith(":") or suffix.startswith("/"):
        return True
    if re.match(r"^\s*(?:am|pm)\b", suffix):
        return True
    if ("platelet" in q or "plt" in q) and re.search(
        r"\b(?:am|pm)\b",
        str(text)[max(0, match.start(1) - 4):match.end(1) + 8].lower(),
    ):
        return True
    return False


def _extract_numeric_from_retrieved(question: str, top_chunks: list[str]) -> str | None:
    import re

    target_patterns = _numeric_target_patterns(question)
    if not target_patterns:
        return None

    target_group = r"(?:%s)" % "|".join(target_patterns)
    number_group = r"([-+]?\d+(?:\.\d+)?)"
    after_target_regex = re.compile(
        rf"{target_group}\s*(?:score|level|value|fraction)?\s*(?:is|was|of)?\s*[-:=]?\s*<?\s*{number_group}\s*%?",
        flags=re.IGNORECASE,
    )
    before_target_regex = re.compile(
        rf"{number_group}\s*%?\s*(?:[-:=]?\s*)?{target_group}",
        flags=re.IGNORECASE,
    )
    allow_before_target = any(
        phrase in str(question).lower()
        for phrase in {"ejection fraction", "left ventricular ejection", "lvef", "chads2", "cha2ds2"}
    )
    candidates = []
    for chunk in top_chunks:
        for match in after_target_regex.finditer(str(chunk)):
            try:
                value = float(match.group(1))
            except ValueError:
                continue
            if _is_spurious_numeric_match(question, str(chunk), match):
                continue
            if _valid_numeric_value(question, value):
                candidates.append(value)
        if allow_before_target:
            for match in before_target_regex.finditer(str(chunk)):
                try:
                    value = float(match.group(1))
                except ValueError:
                    continue
                if _is_spurious_numeric_match(question, str(chunk), match):
                    continue
                if _valid_numeric_value(question, value):
                    candidates.append(value)
    if not candidates:
        return None
    value = min(candidates) if _is_lowest_question(question) else max(candidates)
    return _format_numeric_value(value)


def _langextract_input_from_row(row) -> str:
    return "\n\n---\n\n".join(_parse_top_chunks(row.get("top_chunks", "")))


def main():
    cfg = load_config()
    retrieval_df = pd.read_csv(cfg["paths"]["retrieval_path"], keep_default_na=False)
    primekg_df = pd.read_csv(cfg["paths"]["primekg_path"], keep_default_na=False)
    max_rows = cfg.get("dataset", {}).get("max_rows")
    if isinstance(max_rows, int) and max_rows > 0:
        retrieval_df = retrieval_df.head(max_rows)
        primekg_df = primekg_df.head(max_rows)

    merge_keys = ["note_id", "hadm_id", "question"]
    retrieval_df = _normalize_merge_keys(retrieval_df, merge_keys)
    primekg_df = _normalize_merge_keys(primekg_df, merge_keys)
    merged_df = retrieval_df.merge(
        primekg_df[merge_keys + ["evidence"]],
        on=merge_keys,
        how="left",
        validate="one_to_one",
    )
    if len(merged_df) != len(retrieval_df):
        raise ValueError("Merged answer-generation dataframe has unexpected row count")

    langextract_cache = _load_disk_cache()

    llm = VLLMClient(model=cfg["models"]["llm_model_name"])
    answer_generator = AnswerGenerator(llm, cfg.get("generation", {}), cfg.get("dataset", {}))
    total_rows = len(merged_df)
    save_every = 25

    try:
        requests.get(f"{llm.base_url}/models", timeout=10).raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Local vLLM server is not reachable at {llm.base_url}. "
            "Start the server first, then rerun `python -m src.pipeline.save_answer_results`."
        ) from exc

    rows = []
    print(f"[answer] Processing {total_rows} rows...")
    for idx, retrieval_row in merged_df.iterrows():
        question = str(retrieval_row.get("question", "")).strip()
        qtype = str(retrieval_row.get("question_type", "yes")).strip().lower() or "yes"
        sub_queries = _row_sub_queries(retrieval_row)
        top_chunks = _parse_top_chunks(retrieval_row.get("top_chunks", ""))
        primekg_evidence = _filter_primekg_evidence(
            question,
            sub_queries,
            top_chunks,
            retrieval_row.get("evidence", ""),
        )

        facts = langextract_cache.get(
            _cache_key(
                _langextract_input_from_row(retrieval_row),
                question=question or None,
                sub_queries=sub_queries,
                question_type=qtype,
            ),
            {},
        )
        answer_facts = {
            "diseases": facts.get("primekg_diseases", facts.get("diseases", [])),
            "drugs": facts.get("primekg_drugs", facts.get("drugs", [])),
        }

        raw_prediction = ""
        prediction = _extract_numeric_from_retrieved(question, top_chunks) if qtype == "numeric" else None
        if prediction is None:
            raw_prediction, prediction = answer_generator.generate_with_raw(
                qtype=qtype,
                question=question,
                sub_queries=sub_queries,
                top_chunks=top_chunks,
                facts=answer_facts,
                primekg_evidence=primekg_evidence,
            )

        rows.append(
            {
                "note_id": retrieval_row["note_id"],
                "hadm_id": retrieval_row["hadm_id"],
                "question_type": retrieval_row["question_type"],
                "question": retrieval_row["question"],
                "raw_prediction": raw_prediction,
                "prediction": prediction,
            }
        )

        if (idx + 1) % save_every == 0 or (idx + 1) == total_rows:
            save_csv(rows, cfg["paths"]["pred_path"])
            print(f"[answer] {idx + 1}/{total_rows} completed")

    save_csv(rows, cfg["paths"]["pred_path"])
    print(f"[answer] Saved predictions to {cfg['paths']['pred_path']}")
    metrics = evaluate_predictions(cfg["paths"]["pred_path"])
    print(f"[answer] Saved metrics to {cfg['paths']['metric_path']}")
    print(f"[answer] Saved errors to {cfg['paths']['err_path']}")
    print(metrics)


if __name__ == "__main__":
    main()