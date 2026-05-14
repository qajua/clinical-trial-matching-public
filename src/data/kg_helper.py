import os
import pandas as pd
import re

DEFAULT_HINTS = {
    "afib": [
        "atrial fibrillation may be written as afib, a-fib, or atrial fib",
        "paroxysmal, persistent, and permanent are common afib subtypes"
    ],
    "prior_stroke": [
        "stroke may also appear as CVA or cerebrovascular accident",
        "transient ischemic attack may appear as TIA"
    ],
    "arterial_hypertension": [
        "arterial hypertension may appear as hypertension or HTN"
    ],
    "t2d": [
        "type 2 diabetes may appear as diabetes, DM2, T2D, or T2DM"
    ],
    "chads2": [
        "CHADS2 is a stroke risk score used in atrial fibrillation"
    ]
}

CRITERION_TERMS = {
    "afib": ["atrial fibrillation", "afib", "a-fib", "atrial fib"],
    "prior_stroke": ["stroke", "tia", "transient ischemic attack", "cva", "cerebrovascular accident"],
    "arterial_hypertension": ["hypertension", "htn", "high blood pressure"],
    "t2d": ["type 2 diabetes", "diabetes mellitus", "diabetes", "t2d", "t2dm", "dm2"],
    "chads2": ["chads2", "stroke risk", "atrial fibrillation"]
}

_KG_CACHE = None

def _safe_lower(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

def _load_kg(path: str):
    global _KG_CACHE
    if _KG_CACHE is not None:
        return _KG_CACHE

    if not path or not os.path.exists(path):
        _KG_CACHE = None
        return None

    try:
        df = pd.read_csv(path, dtype=str, low_memory=False)
        _KG_CACHE = df
        return df
    except Exception as e:
        print(f"Failed to load PrimeKG kg.csv: {e}")
        _KG_CACHE = None
        return None

def _find_text_columns(df: pd.DataFrame):
    preferred = [
        "x_name", "y_name", "display_relation",
        "x_type", "y_type", "relation",
        "source", "target"
    ]
    cols = [c for c in preferred if c in df.columns]
    if cols:
        return cols

    text_cols = []
    for c in df.columns:
        if str(df[c].dtype) == "object":
            text_cols.append(c)
    return text_cols[:6]

def _whole_word_match(term: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(term.lower())}\b", text.lower()) is not None

def _search_kg_rows(df: pd.DataFrame, terms: list[str], max_rows: int = 5):
    text_cols = _find_text_columns(df)
    if not text_cols:
        return []

    matched_rows = []

    for _, row in df.iterrows():
        joined = " | ".join(_safe_lower(row[c]) for c in text_cols)

        x_type = _safe_lower(row["x_type"]) if "x_type" in row else ""
        y_type = _safe_lower(row["y_type"]) if "y_type" in row else ""

        if "gene/protein" in x_type or "gene/protein" in y_type:
            continue

        score = 0
        for term in terms:
            if _whole_word_match(term, joined):
                score += 1

        if score > 0:
            if "disease" in x_type:
                score += 2
            if "disease" in y_type:
                score += 2

            matched_rows.append((score, row))

    matched_rows.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in matched_rows[:max_rows]]

def _row_to_hint(row: pd.Series):
    candidates = []
    for key in ["x_name", "x_type", "relation", "display_relation", "y_name", "y_type"]:
        if key in row and pd.notna(row[key]):
            candidates.append(str(row[key]).strip())

    if candidates:
        return " | ".join(candidates)

    values = []
    for v in row.values[:6]:
        if pd.notna(v):
            values.append(str(v).strip())
    return " | ".join(values)

def get_primekg_hints(criterion: str, path: str | None = None) -> list[str]:
    criterion = str(criterion).lower()
    terms = CRITERION_TERMS.get(criterion, [criterion])

    df = _load_kg(path) if path else None
    if df is not None:
        rows = _search_kg_rows(df, terms, max_rows=5)
        hints = [_row_to_hint(row) for row in rows if _row_to_hint(row)]
        if hints:
            return hints

    return DEFAULT_HINTS.get(criterion, [])