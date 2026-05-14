import csv
import json
import os
import re
from functools import lru_cache
from typing import Dict, List

from src.data.umls_client import get_umls_client


COMMON_DISEASE_ALIASES = {
    "af": "atrial fibrillation",
    "a fib": "atrial fibrillation",
    "a-fib": "atrial fibrillation",
    "afib": "atrial fibrillation",
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "dm2": "type 2 diabetes mellitus",
    "t2d": "type 2 diabetes mellitus",
    "t2dm": "type 2 diabetes mellitus",
    "type 2 diabetes": "type 2 diabetes mellitus",
    "cvd": "cardiovascular disease",
    "cad": "coronary artery disease",
    "chf": "congestive heart failure",
    "ckd": "chronic kidney disease",
    "copd": "chronic obstructive pulmonary disease",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "mi": "myocardial infarction",
}


COMMON_DRUG_ALIASES = {
    "eliquis": "apixaban",
    "xarelto": "rivaroxaban",
    "coumadin": "warfarin",
    "glucophage": "metformin",
    "lasix": "furosemide",
}

DEFAULT_UMLS_LEXICON_PATH = os.getenv("UMLS_LEXICON_PATH", "data/umls/umls_lexicon.tsv")


def normalize_text(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    return re.sub(r"\s+", " ", normalized).strip()


@lru_cache(maxsize=1)
def _load_umls_lexicon() -> Dict[str, List[dict]]:
    path = DEFAULT_UMLS_LEXICON_PATH
    if not path or not os.path.exists(path):
        return {}

    lexicon: Dict[str, List[dict]] = {}
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            rows = data.get("entries", [])
        else:
            rows = []
    else:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)

    for row in rows:
        alias = normalize_text(row.get("alias", ""))
        canonical = str(row.get("canonical_name", "")).strip()
        entity_type = normalize_text(row.get("entity_type", ""))
        cui = str(row.get("cui", "")).strip()
        if not alias or not canonical:
            continue
        lexicon.setdefault(alias, []).append(
            {
                "alias": alias,
                "canonical_name": canonical,
                "entity_type": entity_type,
                "cui": cui,
            }
        )
    return lexicon


def align_entity_umls(name: str, entity_type: str, query_context: str = "") -> dict | None:
    normalized = normalize_text(name)
    if not normalized:
        return None

    candidates = _load_umls_lexicon().get(normalized, [])
    if not candidates:
        return None

    target_type = normalize_text(entity_type)
    filtered = [
        item for item in candidates
        if not item.get("entity_type") or item.get("entity_type") == target_type
    ] or candidates

    query_tokens = set(normalize_text(query_context).split())

    def score(item: dict) -> tuple[float, int]:
        canonical_tokens = set(normalize_text(item.get("canonical_name", "")).split())
        overlap = len(query_tokens & canonical_tokens)
        has_type = int(bool(item.get("entity_type")))
        return (float(overlap), has_type)

    best = max(filtered, key=score)
    return {
        "name": str(best.get("canonical_name", "")).strip(),
        "cui": str(best.get("cui", "")).strip(),
        "entity_type": str(best.get("entity_type", "")).strip(),
        "source": "umls",
    }


def align_entity_umls_rest(name: str, entity_type: str, query_context: str = "") -> dict | None:
    client = get_umls_client()
    if not client.enabled:
        return None
    return client.best_match(
        entity_name=name,
        entity_type=entity_type,
        query_context=query_context,
    )


def normalize_entity_name(name: str, entity_type: str, query_context: str = "") -> str:
    normalized = normalize_text(name)
    if not normalized:
        return ""

    umls_match = align_entity_umls(name, entity_type, query_context=query_context)
    if umls_match and umls_match.get("name"):
        return normalize_text(umls_match["name"])

    umls_rest_match = align_entity_umls_rest(name, entity_type, query_context=query_context)
    if umls_rest_match and umls_rest_match.get("name"):
        return normalize_text(umls_rest_match["name"])

    if entity_type == "disease":
        return COMMON_DISEASE_ALIASES.get(normalized, normalized)
    if entity_type == "drug":
        return COMMON_DRUG_ALIASES.get(normalized, normalized)
    return normalized


def _dedup(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = normalize_text(item)
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def normalize_entities(diseases: List[str], drugs: List[str]) -> Dict[str, List[str]]:
    diseases_raw = [str(x).strip() for x in diseases if str(x).strip()]
    drugs_raw = [str(x).strip() for x in drugs if str(x).strip()]

    diseases_norm = _dedup(
        [normalize_entity_name(name, "disease") for name in diseases_raw if normalize_entity_name(name, "disease")]
    )
    drugs_norm = _dedup(
        [normalize_entity_name(name, "drug") for name in drugs_raw if normalize_entity_name(name, "drug")]
    )

    return {
        "diseases_raw": _dedup(diseases_raw),
        "drugs_raw": _dedup(drugs_raw),
        "diseases_norm": diseases_norm,
        "drugs_norm": drugs_norm,
    }