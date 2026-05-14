import json
import re

import pandas as pd
import requests

from src.config import load_config
from src.llm.vllm_client import VLLMClient
from src.pipeline.evaluator import _attach_ground_truth, norm_boolean, norm_numeric
from src.utils.io import save_csv, save_json


def _parse_top_chunks(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.split(" ||| ") if part.strip()]


def _load_query_optimizations(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


YESNO_SYSTEM_PROMPT = """
You are a clinical note evidence judge.

You will be given:
1. A yes/no question
2. Short sub-queries that clarify the target condition
3. Retrieved evidence chunks from the note

Rules:
- Use only the retrieved evidence chunks as evidence
- Output format:
  - First line must be exactly: Yes or No
  - Do not put anything before the first-line answer
- Answer Yes only if the note contains direct and explicit evidence
- If the condition is absent, denied, not mentioned, historical-only, planned-only, or too indirect, answer No
- Prefer No over Yes when the evidence is weak or indirect
- If the evidence is too incomplete to judge, answer No
- Return only valid JSON with exactly one key:
  {"answer":"Yes"}
  or
  {"answer":"No"}
- Do not explain
"""


NUMERIC_SYSTEM_PROMPT = """
You are a clinical note value extractor.

You will be given:
1. A numeric question
2. Short sub-queries that clarify the target value
3. Retrieved evidence chunks from the note

Rules:
- Use only the retrieved evidence chunks as evidence
- Answer only the numeric value or NA
- Return a bare number such as 8.1, 100, 55, or NA
- If a numeric value is present in the evidence, return the value instead of NA
- Do not return units, bullets, list numbering, or any explanation
"""


def _build_retrieval_prompt(question: str, sub_queries: list[str], top_chunks: list[str]) -> str:
    return f"""Question:
{question.strip() or "None"}

Decomposed sub-queries:
{"\n".join(sub_queries) if sub_queries else "None"}

Retrieved evidence chunks:
{"\n\n---\n\n".join(top_chunks) if top_chunks else "None"}
"""


def _extract_yesno_from_json(text: str) -> str | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    candidates = [raw]
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            answer = str(parsed.get("answer", "")).strip().lower()
            if answer in {"yes", "no"}:
                return answer.capitalize()
    return None


def _postprocess_yesno(prediction: str) -> str:
    text = str(prediction).strip()
    if not text:
        return "No"

    first = text.splitlines()[0].strip().lower()
    first_token = re.sub(r"[^a-z]", "", first.split()[0]) if first.split() else ""
    if first_token in {"yes", "no"}:
        return first_token.capitalize()
    return "No"


def _postprocess_numeric(prediction: str) -> str:
    text = str(prediction).strip()
    if not text:
        return "NA"

    lowered = text.lower()
    if lowered == "na" or "not stated" in lowered or "not reported" in lowered:
        return "NA"

    cleaned = "\n".join(
        re.sub(r"^\s*\d+[\.\)]\s*", "", re.sub(r"^\s*[-*]\s*", "", line))
        for line in text.splitlines()
    ).strip()

    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned):
        return cleaned

    matches = re.findall(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?", cleaned)
    if not matches:
        return "NA"
    return matches[-1]


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


def _is_spurious_numeric_match(question: str, text: str, match: re.Match) -> bool:
    q = str(question).lower()
    suffix = str(text)[match.end(1):match.end(1) + 8].lower()
    if suffix.startswith(":") or suffix.startswith("/"):
        return True
    if re.match(r"^\s*(?:am|pm)\b", suffix):
        return True
    if ("platelet" in q or "plt" in q) and re.search(r"\b(?:am|pm)\b", str(text)[max(0, match.start(1) - 4):match.end(1) + 8].lower()):
        return True
    return False


def _extract_numeric_from_retrieved(question: str, top_chunks: list[str]) -> str | None:
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
    candidates: list[float] = []
    allow_before_target = any(
        phrase in str(question).lower()
        for phrase in {"ejection fraction", "left ventricular ejection", "lvef", "chads2", "cha2ds2"}
    )

    for chunk in top_chunks:
        text = str(chunk)
        for match in after_target_regex.finditer(text):
            try:
                value = float(match.group(1))
            except ValueError:
                continue
            if _is_spurious_numeric_match(question, text, match):
                continue
            if _valid_numeric_value(question, value):
                candidates.append(value)

        if allow_before_target:
            for match in before_target_regex.finditer(text):
                try:
                    value = float(match.group(1))
                except ValueError:
                    continue
                if _is_spurious_numeric_match(question, text, match):
                    continue
                if _valid_numeric_value(question, value):
                    candidates.append(value)

    if not candidates:
        return None

    value = min(candidates) if _is_lowest_question(question) else max(candidates)
    return _format_numeric_value(value)


def _evaluate_and_save(pred_path: str, metric_path: str, err_path: str) -> dict:
    cfg = load_config()
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

    save_json(metrics, metric_path)
    err_rows = df[df["correct"] == 0][
        ["note_id", "hadm_id", "question_type", "question", "ground_truth", "prediction", "gold_norm", "pred_norm"]
    ].to_dict(orient="records")
    save_csv(
        err_rows,
        err_path,
        fieldnames=["note_id", "hadm_id", "question_type", "question", "ground_truth", "prediction", "gold_norm", "pred_norm"],
    )
    return metrics


def main():
    cfg = load_config()
    retrieval_df = pd.read_csv(cfg["paths"]["retrieval_path"])
    qopt_map = _load_query_optimizations("outputs/query_optimizations.json")
    total_rows = len(retrieval_df)
    save_every = 25

    pred_path = "outputs/ds_pred_retrieval_qopt.csv"
    metric_path = "outputs/ds_metric_retrieval_qopt.json"
    err_path = "outputs/ds_err_retrieval_qopt.csv"

    llm = VLLMClient(model=cfg["models"]["llm_model_name"])
    temperature = cfg.get("generation", {}).get("temperature", 0.0)
    max_tokens = cfg.get("dataset", {}).get("answer_max_tokens", 32)

    try:
        requests.get(f"{llm.base_url}/models", timeout=10).raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Local vLLM server is not reachable at {llm.base_url}. "
            "Start the server first, then rerun `python -m src.pipeline.save_answer_results_retrieval_qopt`."
        ) from exc

    rows = []
    print(f"[answer-retrieval-qopt] Processing {total_rows} rows...")

    for idx, (_, retrieval_row) in enumerate(retrieval_df.iterrows(), start=1):
        question = str(retrieval_row.get("question", "")).strip()
        qtype = str(retrieval_row.get("question_type", "yes")).strip().lower() or "yes"
        sub_queries = qopt_map.get(question, {}).get("sub_queries", [])
        if not isinstance(sub_queries, list):
            sub_queries = []

        top_chunks = _parse_top_chunks(retrieval_row.get("top_chunks", ""))
        prediction = None
        raw_prediction = ""
        if qtype == "numeric":
            prediction = _extract_numeric_from_retrieved(question, top_chunks)

        if prediction is None:
            user_prompt = _build_retrieval_prompt(question=question, sub_queries=sub_queries, top_chunks=top_chunks)
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
            save_csv(rows, pred_path)
            print(f"[answer-retrieval-qopt] {idx}/{total_rows} completed")

    save_csv(rows, pred_path)
    metrics = _evaluate_and_save(pred_path, metric_path, err_path)

    print(f"[answer-retrieval-qopt] Saved predictions to {pred_path}")
    print(f"[answer-retrieval-qopt] Saved metrics to {metric_path}")
    print(f"[answer-retrieval-qopt] Saved errors to {err_path}")
    print(metrics)


if __name__ == "__main__":
    main()