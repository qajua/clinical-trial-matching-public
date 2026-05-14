from decimal import Decimal, InvalidOperation

import pandas as pd

from src.config import load_config
from src.utils.io import save_csv, save_json


MERGE_KEYS = ["note_id", "hadm_id", "question_type", "question"]


def _norm_common(value):
    if pd.isna(value):
        return "NA"
    text = str(value).strip()
    if not text:
        return "NA"
    lowered = text.lower()
    if lowered in {"nan", "none", "null"}:
        return "NA"
    return text


def norm_boolean(value):
    text = _norm_common(value)
    lowered = text.lower()
    if lowered in {"yes", "y", "true"}:
        return "Yes"
    if lowered in {"no", "n", "false"}:
        return "No"
    if lowered == "na":
        return "NA"
    return text


def norm_numeric(value):
    text = _norm_common(value)
    if text.upper() == "NA":
        return "NA"
    try:
        number = Decimal(text)
        normalized = format(number.normalize(), "f")
        if "." in normalized:
            normalized = normalized.rstrip("0").rstrip(".")
        return normalized if normalized else "0"
    except (InvalidOperation, ValueError):
        return text


def _attach_ground_truth(pred_df: pd.DataFrame, dataset_df: pd.DataFrame) -> pd.DataFrame:
    gold_df = dataset_df[MERGE_KEYS + ["answer"]].rename(columns={"answer": "ground_truth"})
    merged = pred_df.merge(gold_df, on=MERGE_KEYS, how="left", validate="many_to_one")
    return merged


def evaluate_predictions(pred_path: str | None = None) -> dict:
    cfg = load_config()
    pred_path = pred_path or cfg["paths"]["pred_path"]

    pred_df = pd.read_csv(pred_path)
    dataset_df = pd.read_csv(cfg["paths"]["dataset_path"])
    df = _attach_ground_truth(pred_df, dataset_df)

    yes_mask = df["question_type"].astype(str).str.strip().str.lower() == "yes"
    numeric_mask = df["question_type"].astype(str).str.strip().str.lower() == "numeric"

    df["gold_norm"] = "NA"
    df["pred_norm"] = "NA"
    df.loc[yes_mask, "gold_norm"] = df.loc[yes_mask, "ground_truth"].apply(norm_boolean)
    df.loc[yes_mask, "pred_norm"] = df.loc[yes_mask, "prediction"].apply(norm_boolean)
    df.loc[numeric_mask, "gold_norm"] = df.loc[numeric_mask, "ground_truth"].apply(norm_numeric)
    df.loc[numeric_mask, "pred_norm"] = df.loc[numeric_mask, "prediction"].apply(norm_numeric)
    df["correct"] = (df["gold_norm"] == df["pred_norm"]).astype(int)

    boolean_df = df[yes_mask].copy()
    numeric_df = df[numeric_mask].copy()

    metrics = {
        "overall_accuracy": float(df["correct"].mean()) if len(df) else 0.0,
        "boolean_accuracy": float(boolean_df["correct"].mean()) if len(boolean_df) else 0.0,
        "numeric_accuracy": float(numeric_df["correct"].mean()) if len(numeric_df) else 0.0,
        "n_samples": int(len(df)),
        "n_boolean": int(len(boolean_df)),
        "n_numeric": int(len(numeric_df)),
        "by_question_type": (
            df.groupby("question_type")["correct"]
            .mean()
            .reset_index()
            .rename(columns={"correct": "accuracy"})
            .to_dict(orient="records")
        ),
    }

    save_json(metrics, cfg["paths"]["metric_path"])

    err_rows = df[df["correct"] == 0][
        ["note_id", "hadm_id", "question_type", "question", "ground_truth", "prediction", "gold_norm", "pred_norm"]
    ].to_dict(orient="records")
    save_csv(
        err_rows,
        cfg["paths"]["err_path"],
        fieldnames=["note_id", "hadm_id", "question_type", "question", "ground_truth", "prediction", "gold_norm", "pred_norm"],
    )
    return metrics


def main(question: str | None = None, run_inference: bool = False):
    if question is not None:
        raise ValueError("Evaluation with a user-supplied question is not supported because no ground-truth labels are available.")

    if run_inference:
        from src.pipeline.run_pipeline import main as run_infer
        run_infer(question=None)

    cfg = load_config()
    metrics = evaluate_predictions(cfg["paths"]["pred_path"])
    print("\nEvaluation summary:")
    print(metrics)
    print(f"\nSaved metrics to {cfg['paths']['metric_path']}")
    print(f"Saved errors to {cfg['paths']['err_path']}")


if __name__ == "__main__":
    main()