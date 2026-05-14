from src.config import load_config
from src.data.data_loader import load_ds
from src.retrieval.vector_embedder import Embedder
from src.retrieval.evidence_search import EvidenceSearch
from src.utils.io import save_csv


RETRIEVAL_ONLY_PATH = "outputs/ds_retrieval_only.csv"


def main():
    cfg = load_config()
    df = load_ds(cfg["paths"]["dataset_path"])
    max_rows = cfg["dataset"]["max_rows"]

    if isinstance(max_rows, int) and max_rows > 0:
        df = df.head(max_rows)

    embedder = Embedder(cfg["models"]["embedding_model_name"])
    evidence_search = EvidenceSearch(embedder)

    rows = []
    print(f"[retrieval-only] Processing {len(df)} rows...")

    for _, row in df.iterrows():
        question = str(row["question"])

        top_chunks = evidence_search.search_and_rerank(
            note_id=str(row["note_id"]),
            note_text=str(row["text"]),
            retrieval_query=question,
            retrieval_cfg=cfg["retrieval"],
            question=question,
            question_type=str(row["question_type"]),
        )

        result_row = {
            "note_id": row["note_id"],
            "hadm_id": row["hadm_id"],
            "question_type": row["question_type"],
            "question": question,
            "sub_queries": "",
            "top_chunks": " ||| ".join(top_chunks),
        }

        rows.append(result_row)

    save_csv(rows, RETRIEVAL_ONLY_PATH)
    print(f"[retrieval-only] Saved retrieval results to {RETRIEVAL_ONLY_PATH}")


if __name__ == "__main__":
    main()