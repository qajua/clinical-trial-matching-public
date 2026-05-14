import os
import json
import hashlib
import re
from dotenv import load_dotenv

load_dotenv()

from typing import Any, Dict, List, Sequence

from src.data.entity_normalizer import normalize_entity_name, normalize_text
from src.llm.vllm_client import VLLMClient

LOCAL_EXTRACT_SYSTEM_PROMPT = """
Extract disease/drug entities from clinical note evidence and decide whether PrimeKG seeds are needed.

Return ONLY JSON. No thinking, markdown, explanation, or commentary.

Input has QUESTION, SUB-QUERIES, QUESTION TYPE, and TOP CHUNKS.

Rules:
- Extract only disease and drug entities.
- Do not extract phenotypes, labs, procedures, anatomy, pathology descriptors, numeric thresholds, or vague findings.
- Extract diseases/drugs from the evidence without forcing an arbitrary count limit.
- primekg_diseases and primekg_drugs must be STRICT subsets of diseases and drugs.
- Use QUESTION/SUB-QUERIES only to decide whether PrimeKG seeds are needed.
- Put an entity in primekg_* only if it would be useful as a minimal PrimeKG seed for answering the question.
- If PrimeKG is not needed, return empty primekg_diseases and primekg_drugs.
- Do not put unrelated background comorbidities, routine/supportive meds, antibiotics,
  chemotherapy regimens, or incidental diseases into primekg_* merely because they appear.
- Numeric/lab questions often do not need PrimeKG; leave primekg_* empty unless disease/drug graph context is truly needed.

Return ONLY valid JSON with exactly these keys:
{
  "diseases": [],
  "drugs": [],
  "primekg_diseases": [],
  "primekg_drugs": []
}

Example:
{
  "diseases": ["Diabetes", "Hypertension", "atrial fibrillation", "Prior stroke"],
  "drugs": ["Apixaban"],
  "primekg_diseases": ["atrial fibrillation"],
  "primekg_drugs": ["Apixaban"]
}
"""

LOCAL_EXTRACT_MODEL_ENV = "LANGEXTRACT_LOCAL_MODEL"
LOCAL_EXTRACT_BASE_URL_ENV = "LANGEXTRACT_LOCAL_BASE_URL"
DEFAULT_LOCAL_EXTRACT_MODEL = "Qwen/Qwen3.5-27B"
_CACHE_VERSION = "local-disease-drug-json-v9-model-decides-primekg"
_LOCAL_EXTRACT_MAX_TOKENS = (512, 384, 256)

_CACHE_DIR = "outputs"
_CACHE_PATH = os.path.join(_CACHE_DIR, "langextract_entities.json")
_MEMORY_CACHE: Dict[str, Dict[str, Any]] = {}


def _empty_result() -> Dict[str, Any]:
    return {
        "diseases": [],
        "drugs": [],
        "primekg_diseases": [],
        "primekg_drugs": [],
    }


def _clean_entity(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip(" -:;,.")


def _dedup(items: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        cleaned = _clean_entity(item)
        key = normalize_text(cleaned)
        if key and key not in seen:
            seen.add(key)
            out.append(cleaned)
    return out


def _canonicalize_entities(items: Sequence[str], entity_type: str) -> List[str]:
    canonical = []
    seen = set()
    for item in items:
        cleaned = _clean_entity(item)
        normalized = normalize_entity_name(cleaned, entity_type)
        normalized = _clean_entity(normalized)
        key = normalize_text(normalized)
        if key and key not in seen:
            seen.add(key)
            canonical.append(normalized)
    return canonical


def _entity_subset_keys(item: str, entity_type: str) -> set[str]:
    normalized = normalize_text(item)
    aliased = normalize_text(normalize_entity_name(item, entity_type))
    return {key for key in (normalized, aliased) if key}


def _keep_entity_subset(items: Sequence[str], allowed: Sequence[str], entity_type: str) -> List[str]:
    allowed_keys = set()
    for item in allowed:
        allowed_keys.update(_entity_subset_keys(item, entity_type))

    return _dedup(
        item
        for item in items
        if _entity_subset_keys(item, entity_type) & allowed_keys
    )


def _normalize_result(
    data: Dict[str, Any] | None,
    question_type: str | None = None,
    question: str | None = None,
    sub_queries: Sequence[str] | str | None = None,
) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return _empty_result()

    diseases = data.get("diseases", data.get("primary_diseases", []))
    drugs = data.get("drugs", data.get("primary_drugs", []))
    primekg_diseases = data.get("primekg_diseases", [])
    primekg_drugs = data.get("primekg_drugs", [])

    diseases = diseases if isinstance(diseases, list) else []
    drugs = drugs if isinstance(drugs, list) else []
    primekg_diseases = primekg_diseases if isinstance(primekg_diseases, list) else []
    primekg_drugs = primekg_drugs if isinstance(primekg_drugs, list) else []

    diseases = _canonicalize_entities(diseases, "disease")
    drugs = _canonicalize_entities(drugs, "drug")
    primekg_diseases = _canonicalize_entities(
        _keep_entity_subset(primekg_diseases, diseases, "disease"),
        "disease",
    )
    primekg_drugs = _canonicalize_entities(
        _keep_entity_subset(primekg_drugs, drugs, "drug"),
        "drug",
    )

    return {
        "diseases": diseases,
        "drugs": drugs,
        "primekg_diseases": primekg_diseases,
        "primekg_drugs": primekg_drugs,
    }


def _format_input(
    text: str,
    question: str | None = None,
    sub_queries: Sequence[str] | str | None = None,
    question_type: str | None = None,
) -> str:
    if isinstance(sub_queries, str):
        sub_query_lines = [x.strip() for x in re.split(r"\s*\|\s*|\n+", sub_queries) if x.strip()]
    else:
        sub_query_lines = [str(x).strip() for x in (sub_queries or []) if str(x).strip()]

    sub_query_text = "\n".join(f"- {x}" for x in sub_query_lines) if sub_query_lines else "-"
    return "\n".join(
        [
            "QUESTION:",
            str(question or "").strip() or "-",
            "SUB-QUERIES:",
            sub_query_text,
            "QUESTION TYPE:",
            str(question_type or "").strip() or "-",
            "TOP CHUNKS:",
            str(text).strip(),
        ]
    )


def _get_local_llm(llm: VLLMClient | None = None) -> VLLMClient:
    if llm is not None:
        return llm
    return VLLMClient(
        model=os.getenv(LOCAL_EXTRACT_MODEL_ENV, DEFAULT_LOCAL_EXTRACT_MODEL),
        base_url=os.getenv(LOCAL_EXTRACT_BASE_URL_ENV, "http://localhost:8000/v1"),
    )


def _strip_json_fence(raw: str) -> str:
    text = str(raw).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_json_object(raw: str) -> Dict[str, Any]:
    text = _strip_json_fence(raw)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def _cache_key(text: str, question: str | None = None, sub_queries: Sequence[str] | str | None = None, question_type: str | None = None) -> str:
    normalized = " ".join(
        f"{_CACHE_VERSION}\n{_format_input(text, question, sub_queries, question_type)}".split()
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _load_disk_cache() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(_CACHE_PATH):
        return {}

    try:
        with open(_CACHE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {}
        return {str(k): v for k, v in raw.items() if isinstance(v, dict)}
    except Exception as e:
        print(f"Failed to load LangExtract cache, ignoring cache file: {e}")
        return {}


def _save_disk_cache(cache: Dict[str, Dict[str, Any]]) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def extract_facts(
    text: str,
    question: str | None = None,
    sub_queries: Sequence[str] | str | None = None,
    question_type: str | None = None,
    llm: VLLMClient | None = None,
) -> Dict[str, Any]:
    """
    Extract disease/drug entities from question + retrieved evidence.

    The returned dict keeps all extracted ``diseases`` and ``drugs`` plus the subset
    selected for PrimeKG in ``primekg_diseases`` and ``primekg_drugs``.
    Results are cached on disk so repeated runs do not re-call LangExtract for the same text.
    """
    empty_result = _empty_result()
    
    if not text or not text.strip():
        return empty_result

    key = _cache_key(text, question=question, sub_queries=sub_queries, question_type=question_type)
    if key in _MEMORY_CACHE:
        return _MEMORY_CACHE[key]

    disk_cache = _load_disk_cache()
    if key in disk_cache:
        cached = _normalize_result(
            disk_cache[key],
            question_type=question_type,
            question=question,
            sub_queries=sub_queries,
        )
        _MEMORY_CACHE[key] = cached
        return cached

    try:
        extraction_input = _format_input(
            text,
            question=question,
            sub_queries=sub_queries,
            question_type=question_type,
        )
        local_llm = _get_local_llm(llm)
        user_prompt = "Return only the JSON object for this input.\n\n" + extraction_input
        last_error = None
        raw = ""
        for max_tokens in _LOCAL_EXTRACT_MAX_TOKENS:
            try:
                raw = local_llm.generate(
                    system_prompt=LOCAL_EXTRACT_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                break
            except Exception as e:
                last_error = e
                if "maximum input length" not in str(e) and "context length" not in str(e):
                    raise
        else:
            raise last_error or RuntimeError("langextract generation failed")

        entities = _parse_json_object(raw)
        entities = _normalize_result(
            entities,
            question_type=question_type,
            question=question,
            sub_queries=sub_queries,
        )
        _MEMORY_CACHE[key] = entities
        disk_cache[key] = entities
        _save_disk_cache(disk_cache)
        return entities

    except Exception as e:
        print(f"langextract failed, returning empty dict: {e}")
        return empty_result