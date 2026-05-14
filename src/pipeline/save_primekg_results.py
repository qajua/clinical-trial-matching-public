import pandas as pd

from src.config import load_config
from src.data.primekg import PrimeKGHelper
from src.pipeline.langextract import _cache_key, _load_disk_cache
from src.pipeline.query_optimizer import optimize_query
from src.utils.io import save_csv


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


def _langextract_input_from_row(row) -> str:
    chunk_text = str(row.get("top_chunks", "")).strip()
    return "\n\n---\n\n".join(
        part.strip()
        for part in chunk_text.split(" ||| ")
        if part.strip()
    )


def main():
    cfg = load_config()
    df = pd.read_csv(cfg["paths"]["retrieval_path"])
    max_rows = cfg.get("dataset", {}).get("max_rows")
    if isinstance(max_rows, int) and max_rows > 0:
        df = df.head(max_rows)

    langextract_cache = _load_disk_cache()

    primekg = PrimeKGHelper(
        kg_csv_path=cfg["paths"]["primekg_hints_path"],
    )

    rows = []
    print(f"[primekg] Processing {len(df)} rows...")

    for _, row in df.iterrows():
        question = str(row.get("question", "")).strip()
        sub_queries = _row_sub_queries(row)
        question_type = str(row.get("question_type", "")).strip().lower() or None
        langextract_input = _langextract_input_from_row(row)
        facts = langextract_cache.get(
            _cache_key(
                langextract_input,
                question=question or None,
                sub_queries=sub_queries,
                question_type=question_type,
            ),
            {},
        )
        primekg_facts = {
            "diseases": facts.get("primekg_diseases", []),
            "drugs": facts.get("primekg_drugs", []),
        }
        kg_result = primekg.build_primekg_evidence_from_structured_entities(
            entities_dict=primekg_facts,
            question=question,
            sub_queries=sub_queries,
            max_relations=6,
            max_paths=3,
            question_type=question_type,
        )

        rows.append({
            "note_id": row["note_id"],
            "hadm_id": row["hadm_id"],
            "question": question,
            "evidence": kg_result.get("primekg_evidence", ""),
        })

    save_csv(rows, cfg["paths"]["primekg_path"])
    print(f"[primekg] Saved PrimeKG results to {cfg['paths']['primekg_path']}")


if __name__ == "__main__":
    main()