from src.config import load_config
from src.data.data_loader import load_ds
from src.pipeline.query_optimizer import optimize_query
from src.retrieval.vector_embedder import Embedder
from src.retrieval.evidence_search import EvidenceSearch
from src.utils.io import save_csv


def main():
    cfg = load_config()
    df = load_ds(cfg["paths"]["dataset_path"])
    max_rows = cfg["dataset"]["max_rows"]

    if isinstance(max_rows, int) and max_rows > 0:
        df = df.head(max_rows)

    embedder = Embedder(cfg["models"]["embedding_model_name"])
    evidence_search = EvidenceSearch(embedder)

    rows = []
    print(f"[retrieval] Processing {len(df)} rows...")

    for _, row in df.iterrows():
        question = str(row["question"])
        qopt_result = optimize_query(question, llm=None)
        rewritten_query = qopt_result.get("rewritten_query", question)
        sub_queries = qopt_result.get("sub_queries", [])
        if not isinstance(sub_queries, list):
            sub_queries = []
        retrieval_query = qopt_result.get("retrieval_query", rewritten_query)

        top_chunks = evidence_search.search_and_rerank(
            note_id=str(row["note_id"]),
            note_text=str(row["text"]),
            retrieval_query=retrieval_query,
            retrieval_cfg=cfg["retrieval"],
            question=question,
            question_type=str(row["question_type"]),
        )

        result_row = {
            "note_id": row["note_id"],
            "hadm_id": row["hadm_id"],
            "question_type": row["question_type"],
            "question": question,
            "sub_queries": " | ".join(str(item).strip() for item in sub_queries if str(item).strip()),
            "top_chunks": " ||| ".join(top_chunks),
        }

        rows.append(result_row)

    save_csv(rows, cfg["paths"]["retrieval_path"])
    print(f"[retrieval] Saved retrieval results to {cfg['paths']['retrieval_path']}")


if __name__ == "__main__":
    main()