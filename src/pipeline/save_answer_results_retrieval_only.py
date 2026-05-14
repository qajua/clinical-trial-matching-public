import pandas as pd
import requests

from src.config import load_config
from src.llm.vllm_client import VLLMClient
from src.pipeline.save_answer_results_retrieval_qopt import (
    NUMERIC_SYSTEM_PROMPT,
    YESNO_SYSTEM_PROMPT,
    _build_retrieval_prompt,
    _evaluate_and_save,
    _extract_numeric_from_retrieved,
    _extract_yesno_from_json,
    _parse_top_chunks,
    _postprocess_numeric,
    _postprocess_yesno,
)
from src.utils.io import save_csv


RETRIEVAL_ONLY_PATH = "outputs/ds_retrieval_only.csv"
RETRIEVAL_ONLY_PRED_PATH = "outputs/ds_pred_retrieval_only.csv"
RETRIEVAL_ONLY_METRIC_PATH = "outputs/ds_metric_retrieval_only.json"
RETRIEVAL_ONLY_ERR_PATH = "outputs/ds_err_retrieval_only.csv"


def main():
    cfg = load_config()
    retrieval_df = pd.read_csv(RETRIEVAL_ONLY_PATH)
    total_rows = len(retrieval_df)
    save_every = 25

    llm = VLLMClient(model=cfg["models"]["llm_model_name"])
    temperature = cfg.get("generation", {}).get("temperature", 0.0)
    max_tokens = cfg.get("dataset", {}).get("answer_max_tokens", 32)

    try:
        requests.get(f"{llm.base_url}/models", timeout=10).raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Local vLLM server is not reachable at {llm.base_url}. "
            "Start the server first, then rerun `python -m src.pipeline.save_answer_results_retrieval_only`."
        ) from exc

    rows = []
    print(f"[answer-retrieval-only] Processing {total_rows} rows...")

    for idx, (_, retrieval_row) in enumerate(retrieval_df.iterrows(), start=1):
        question = str(retrieval_row.get("question", "")).strip()
        qtype = str(retrieval_row.get("question_type", "yes")).strip().lower() or "yes"
        top_chunks = _parse_top_chunks(retrieval_row.get("top_chunks", ""))
        prediction = None
        raw_prediction = ""

        if qtype == "numeric":
            prediction = _extract_numeric_from_retrieved(question, top_chunks)

        if prediction is None:
            user_prompt = _build_retrieval_prompt(question=question, sub_queries=[], top_chunks=top_chunks)
            if qtype == "yes":
                raw_prediction = llm.generate(
                    system_prompt=YESNO_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=16,
                    response_format={"type": "json_object"},
                ).strip()
                prediction = _extract_yesno_from_json(raw_prediction) or _postprocess_yesno(raw_prediction)
            else:
                raw_prediction = llm.generate(
                    system_prompt=NUMERIC_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ).strip()
                prediction = _postprocess_numeric(raw_prediction)

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

        if idx % save_every == 0 or idx == total_rows:
            save_csv(rows, RETRIEVAL_ONLY_PRED_PATH)
            print(f"[answer-retrieval-only] {idx}/{total_rows} completed")

    save_csv(rows, RETRIEVAL_ONLY_PRED_PATH)
    metrics = _evaluate_and_save(RETRIEVAL_ONLY_PRED_PATH, RETRIEVAL_ONLY_METRIC_PATH, RETRIEVAL_ONLY_ERR_PATH)

    print(f"[answer-retrieval-only] Saved predictions to {RETRIEVAL_ONLY_PRED_PATH}")
    print(f"[answer-retrieval-only] Saved metrics to {RETRIEVAL_ONLY_METRIC_PATH}")
    print(f"[answer-retrieval-only] Saved errors to {RETRIEVAL_ONLY_ERR_PATH}")
    print(metrics)


if __name__ == "__main__":
    main()