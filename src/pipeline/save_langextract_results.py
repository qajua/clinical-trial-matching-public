import pandas as pd

from src.config import load_config
from src.llm.vllm_client import VLLMClient
from src.pipeline.langextract import extract_facts, _CACHE_PATH
from src.pipeline.query_optimizer import optimize_query


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


def _parse_top_chunks(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return []
    return [part.strip() for part in text.split(" ||| ") if part.strip()]


def main():
    cfg = load_config()
    df = pd.read_csv(cfg["paths"]["retrieval_path"])
    max_rows = cfg.get("dataset", {}).get("max_rows")
    if isinstance(max_rows, int) and max_rows > 0:
        df = df.head(max_rows)

    llm = VLLMClient(model=cfg["models"]["llm_model_name"])

    print(f"[langextract] Processing {len(df)} rows...")

    for _, row in df.iterrows():
        langextract_input = "\n\n---\n\n".join(_parse_top_chunks(row.get("top_chunks", "")))
        extract_facts(
            langextract_input,
            question=str(row.get("question", "")).strip() or None,
            sub_queries=_row_sub_queries(row),
            question_type=str(row.get("question_type", "")).strip().lower() or None,
            llm=llm,
        )

    print(f"[langextract] Filled LangExtract cache at {_CACHE_PATH}")


if __name__ == "__main__":
    main()