import hashlib
import os
import csv
import pickle
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Sequence

from src.data.entity_normalizer import (
    align_entity_umls,
    align_entity_umls_rest,
    normalize_entities,
    normalize_entity_name,
    normalize_text,
)


RELATION_PRIORITY = {
    "contraindication": 4.0,
    "indication": 4.0,
    "disease disease": 3.0,
    "phenotype positive": 3.0,
    "phenotype negative": 3.0,
    "drug phenotype": 2.0,
    "drug protein": 1.0,
    "disease protein": 1.0,
}

SHORT_AMBIGUOUS_ENTITY_NAMES = {
    "ar", "eed", "comp", "ins", "nes", "pe", "gi", "net", "gist", "hp",
}

QUESTION_TYPE_RELATIONS = {
    "yes": {"contraindication", "indication", "disease disease", "phenotype positive", "phenotype negative"},
    "numeric": {"contraindication", "indication"},
}


@dataclass
class PrimeKGEntity:
    raw_name: str
    normalized_name: str
    matched_name: str
    entity_type: str
    score: float


@dataclass
class PrimeKGRelation:
    text: str
    score: float
    row: Dict[str, Any]


@dataclass
class PrimeKGPath:
    text: str
    score: float
    rows: List[Dict[str, Any]]


class PrimeKGHelper:
    def __init__(self, kg_csv_path: str):
        self.kg_csv_path = kg_csv_path
        self.cache_dir = "outputs"
        self.index_path = os.path.join(self.cache_dir, "primekg_index.pkl")
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self.rows: List[Dict[str, str]] = []
        self.name_index: Dict[str, List[Dict[str, str]]] = {}
        self.row_index: Dict[str, List[int]] = {}
        self._build_indices()

    def _build_indices(self) -> None:
        if os.path.exists(self.index_path):
            with open(self.index_path, "rb") as f:
                payload = pickle.load(f)
            self.rows = payload.get("rows", [])
            self.name_index = payload.get("name_index", {})
            self.row_index = payload.get("row_index", {})
            return

        allowed_relation_keys = set(RELATION_PRIORITY) | set().union(*QUESTION_TYPE_RELATIONS.values())
        with open(self.kg_csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for raw_row in reader:
                row = {k: (v or "").strip() for k, v in raw_row.items()}
                relation_key = self._relation_key(row)
                if relation_key not in allowed_relation_keys:
                    continue

                row_id = len(self.rows)
                compact_row = {
                    "x_name": row.get("x_name", ""),
                    "x_type": row.get("x_type", ""),
                    "relation": row.get("relation", ""),
                    "display_relation": row.get("display_relation", ""),
                    "y_name": row.get("y_name", ""),
                    "y_type": row.get("y_type", ""),
                }
                self.rows.append(compact_row)

                x_name = compact_row["x_name"]
                y_name = compact_row["y_name"]
                x_type = self._simplify_type(compact_row["x_type"])
                y_type = self._simplify_type(compact_row["y_type"])

                for name, entity_type in ((x_name, x_type), (y_name, y_type)):
                    normalized = normalize_text(name)
                    if not normalized:
                        continue
                    self.name_index.setdefault(normalized, []).append(
                        {"name": name, "entity_type": entity_type}
                    )
                    self.row_index.setdefault(normalized, []).append(row_id)

        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(
                {
                    "rows": self.rows,
                    "name_index": self.name_index,
                    "row_index": self.row_index,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @staticmethod
    def _dedup_keep_order(items: List[str]) -> List[str]:
        seen = set()
        out = []
        for item in items:
            key = normalize_text(item)
            if key and key not in seen:
                seen.add(key)
                out.append(item)
        return out

    @staticmethod
    def _dedup_entities(items: List[PrimeKGEntity]) -> List[PrimeKGEntity]:
        seen = set()
        out = []
        for item in items:
            key = (item.entity_type, normalize_text(item.matched_name), normalize_text(item.raw_name))
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out

    @staticmethod
    def _simplify_type(value: str) -> str:
        text = normalize_text(value)
        if "disease" in text:
            return "disease"
        if "drug" in text:
            return "drug"
        if "phenotype" in text:
            return "phenotype"
        if "protein" in text or "gene" in text:
            return "protein"
        return text or "unknown"

    @staticmethod
    def _relation_key(row: Dict[str, Any]) -> str:
        relation = normalize_text(row.get("relation", ""))
        display = normalize_text(row.get("display_relation", ""))
        combined = f"{relation} {display}".strip()

        if "contraindication" in combined:
            return "contraindication"
        if "indication" in combined or "off label" in combined:
            return "indication"

        x_type = PrimeKGHelper._simplify_type(row.get("x_type", ""))
        y_type = PrimeKGHelper._simplify_type(row.get("y_type", ""))
        if x_type == "disease" and y_type == "disease":
            return "disease disease"
        if "positive" in combined and "phenotype" in combined:
            return "phenotype positive"
        if "negative" in combined and "phenotype" in combined:
            return "phenotype negative"
        if x_type == "drug" and y_type == "phenotype":
            return "drug phenotype"
        if x_type == "drug" and y_type == "protein":
            return "drug protein"
        if x_type == "disease" and y_type == "protein":
            return "disease protein"
        return combined

    def _cache_key(
        self,
        entities_dict: Dict[str, List[str]],
        question_type: str | None,
        question: str = "",
        sub_queries: Sequence[str] | None = None,
    ) -> str:
        normalized = normalize_entities(
            entities_dict.get("diseases", []),
            entities_dict.get("drugs", []),
        )
        payload = str(
            {
                "question": normalize_text(question),
                "sub_queries": [normalize_text(x) for x in (sub_queries or [])],
                "question_type": question_type or "",
                "diseases_norm": normalized["diseases_norm"],
                "drugs_norm": normalized["drugs_norm"],
            }
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _question_focus_keywords(
        self,
        question: str = "",
        sub_queries: Sequence[str] | str | None = None,
    ) -> List[str]:
        if isinstance(sub_queries, str):
            sub_query_items = [x.strip() for x in sub_queries.split("|") if x.strip()]
        else:
            sub_query_items = [str(x).strip() for x in (sub_queries or []) if str(x).strip()]

        text = normalize_text(" ".join([question or ""] + sub_query_items))
        if not text:
            return []

        stopwords = {
            "the", "and", "or", "of", "to", "in", "on", "for", "with", "without", "as",
            "a", "an", "is", "are", "was", "were", "be", "been", "being", "does", "do",
            "did", "patient", "note", "describe", "mentioned", "answer", "having", "have",
            "has", "ever", "current", "currently", "highest", "lowest", "what", "when",
            "where", "which", "type", "score", "value", "number", "if", "no", "yes", "na",
        }
        tokens = [t for t in text.split() if len(t) >= 3 and t not in stopwords]

        keywords: List[str] = []
        for token in tokens:
            keywords.append(token)
            for entity_type in ("disease", "drug"):
                aliased = normalize_text(normalize_entity_name(token, entity_type))
                if aliased:
                    keywords.append(aliased)

        for n in (3, 2):
            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i:i + n])
                keywords.append(phrase)
                disease_alias = normalize_text(normalize_entity_name(phrase, "disease"))
                drug_alias = normalize_text(normalize_entity_name(phrase, "drug"))
                if disease_alias:
                    keywords.append(disease_alias)
                if drug_alias:
                    keywords.append(drug_alias)

        return self._dedup_keep_order(keywords)

    def _focus_overlap_score(self, text: str, focus_keywords: Sequence[str]) -> float:
        normalized = normalize_text(text)
        if not normalized or not focus_keywords:
            return 0.0

        score = 0.0
        text_tokens = set(normalized.split())
        for keyword in focus_keywords:
            key = normalize_text(keyword)
            if not key:
                continue
            if key == normalized:
                score += 4.0
            elif len(key) >= 4 and len(normalized) >= 4 and (key in normalized or normalized in key):
                score += 2.0
            else:
                key_tokens = set(key.split())
                overlap = text_tokens & key_tokens
                if overlap:
                    score += min(1.5, 0.5 * len(overlap))
        return score

    def _entity_focus_score(self, entity: PrimeKGEntity, focus_keywords: Sequence[str]) -> float:
        return max(
            self._focus_overlap_score(entity.raw_name, focus_keywords),
            self._focus_overlap_score(entity.normalized_name, focus_keywords),
            self._focus_overlap_score(normalize_entity_name(entity.raw_name, entity.entity_type), focus_keywords),
            self._focus_overlap_score(entity.matched_name, focus_keywords),
        )

    @staticmethod
    def _entity_name_keys(entities: Sequence[PrimeKGEntity]) -> set[str]:
        keys = set()
        for entity in entities:
            for value in (entity.raw_name, entity.normalized_name, entity.matched_name):
                key = normalize_text(value)
                if key:
                    keys.add(key)
        return keys

    @staticmethod
    def _is_entity_like_keyword(keyword: str, entity_keys: set[str]) -> bool:
        key = normalize_text(keyword)
        if not key:
            return True
        for entity_key in entity_keys:
            if key == entity_key or key in entity_key or entity_key in key:
                return True
        return False

    def _context_focus_keywords(
        self,
        focus_keywords: Sequence[str],
        entities: Sequence[PrimeKGEntity],
    ) -> List[str]:
        entity_keys = self._entity_name_keys(entities)
        return [
            keyword
            for keyword in self._dedup_keep_order(list(focus_keywords))
            if not self._is_entity_like_keyword(keyword, entity_keys)
        ]

    def _relation_is_query_specific(
        self,
        row: Dict[str, Any],
        entities: Sequence[PrimeKGEntity],
        core_entities: Sequence[PrimeKGEntity],
        focus_keywords: Sequence[str],
    ) -> bool:
        relation_key = self._relation_key(row)
        endpoint_keys = self._row_endpoint_keys(row)
        all_entity_keys = self._entity_name_keys(entities)
        core_entity_keys = self._entity_name_keys(core_entities) or all_entity_keys
        context_keywords = self._context_focus_keywords(focus_keywords, entities)

        if not (endpoint_keys & core_entity_keys):
            return False

        row_text = " ".join(
            str(row.get(k, "")) for k in ("x_name", "y_name", "relation", "display_relation", "x_type", "y_type")
        )
        context_overlap = self._focus_overlap_score(row_text, context_keywords)
        seeded_endpoint_count = len(endpoint_keys & all_entity_keys)

        if context_keywords:
            return context_overlap > 0 or (relation_key in {"contraindication", "indication"} and seeded_endpoint_count >= 2)

        if relation_key in {"contraindication", "indication"}:
            return seeded_endpoint_count >= 1

        if relation_key in {"disease disease", "phenotype positive", "phenotype negative"}:
            return seeded_endpoint_count >= 2

        return seeded_endpoint_count >= 2

    def _path_is_query_specific(
        self,
        path: PrimeKGPath,
        entities: Sequence[PrimeKGEntity],
        focus_keywords: Sequence[str],
    ) -> bool:
        all_entity_keys = self._entity_name_keys(entities)
        context_keywords = self._context_focus_keywords(focus_keywords, entities)
        endpoint_keys = set()
        for row in path.rows:
            endpoint_keys.update(self._row_endpoint_keys(row))
        seeded_endpoint_count = len(endpoint_keys & all_entity_keys)

        if context_keywords:
            return self._focus_overlap_score(path.text, context_keywords) > 0 and seeded_endpoint_count >= 1
        return seeded_endpoint_count >= 2

    def _candidate_entity_names(self, raw_name: str, entity_type: str) -> List[str]:
        raw_normalized = normalize_text(raw_name)
        aliased = normalize_text(normalize_entity_name(raw_name, entity_type))
        return self._dedup_keep_order([raw_normalized, aliased])

    def _umls_expanded_names(
        self,
        raw_name: str,
        entity_type: str,
        query_context: str = "",
    ) -> List[str]:
        candidates: List[str] = []

        local_match = align_entity_umls(raw_name, entity_type, query_context=query_context)
        if local_match and local_match.get("name"):
            candidates.append(str(local_match["name"]))

        rest_match = align_entity_umls_rest(raw_name, entity_type, query_context=query_context)
        if rest_match and rest_match.get("name"):
            candidates.append(str(rest_match["name"]))
            for synonym in rest_match.get("synonyms", []) or []:
                synonym = str(synonym).strip()
                if synonym:
                    candidates.append(synonym)

        return self._dedup_keep_order(normalize_text(name) for name in candidates if normalize_text(name))

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return {token for token in normalize_text(text).split() if token}

    def _specificity_bonus(self, raw_name: str, candidate_name: str) -> float:
        raw_norm = normalize_text(raw_name)
        cand_norm = normalize_text(candidate_name)
        if not raw_norm or not cand_norm:
            return 0.0

        raw_tokens = self._token_set(raw_norm)
        cand_tokens = self._token_set(cand_norm)
        if not raw_tokens or not cand_tokens:
            return 0.0

        if raw_norm == cand_norm:
            return 0.8

        if raw_norm in cand_norm and len(cand_tokens) >= len(raw_tokens):
            return 0.45

        if cand_norm in raw_norm and len(raw_tokens) > len(cand_tokens):
            return -0.45

        overlap = len(raw_tokens & cand_tokens)
        if overlap >= max(1, min(len(raw_tokens), len(cand_tokens))):
            if len(cand_tokens) > len(raw_tokens):
                return 0.2
            if len(cand_tokens) < len(raw_tokens):
                return -0.25
        return 0.0

    def _generic_parent_penalty(self, raw_name: str, candidate_name: str, entity_type: str) -> float:
        raw_norm = normalize_text(raw_name)
        cand_norm = normalize_text(candidate_name)
        if not raw_norm or not cand_norm or raw_norm == cand_norm:
            return 0.0

        raw_tokens = self._token_set(raw_norm)
        cand_tokens = self._token_set(cand_norm)
        if not raw_tokens or not cand_tokens:
            return 0.0

        generic_disease_terms = {
            "cancer", "disease", "disorder", "failure", "bleeding", "hemorrhage",
            "injury", "infection", "syndrome", "dysfunction",
        }
        generic_drug_terms = {"drug", "agent", "therapy", "treatment"}
        generic_terms = generic_disease_terms if entity_type == "disease" else generic_drug_terms

        if cand_norm in generic_terms and len(raw_tokens) > 1:
            return 0.9

        if cand_tokens <= raw_tokens and len(cand_tokens) < len(raw_tokens):
            if any(token in generic_terms for token in cand_tokens):
                return 0.55

        if cand_norm in raw_norm and len(cand_tokens) < len(raw_tokens):
            return 0.35

        return 0.0

    def _question_match_bonus(self, candidate_name: str, focus_keywords: Sequence[str]) -> float:
        return 0.25 * self._focus_overlap_score(candidate_name, focus_keywords or [])

    def _question_gate_penalty(
        self,
        raw_name: str,
        candidate_name: str,
        focus_keywords: Sequence[str],
    ) -> float:
        candidate_overlap = self._focus_overlap_score(candidate_name, focus_keywords or [])
        raw_overlap = self._focus_overlap_score(raw_name, focus_keywords or [])
        if raw_overlap >= 1.5 and candidate_overlap <= 0.0:
            return 0.6
        if raw_overlap >= 0.5 and candidate_overlap <= 0.0:
            return 0.3
        return 0.0

    @staticmethod
    def _build_query_context(
        question: str = "",
        sub_queries: Sequence[str] | str | None = None,
    ) -> str:
        if isinstance(sub_queries, str):
            sub_query_items = [x.strip() for x in sub_queries.split("|") if x.strip()]
        else:
            sub_query_items = [str(x).strip() for x in (sub_queries or []) if str(x).strip()]
        return " ".join([str(question or "").strip()] + sub_query_items).strip()

    def _match_entity(
        self,
        raw_name: str,
        entity_type: str,
        focus_keywords: Sequence[str] | None = None,
        query_context: str = "",
    ) -> PrimeKGEntity | None:
        raw_normalized = normalize_text(raw_name)
        candidate_names = self._dedup_keep_order(
            [
                raw_normalized,
                normalize_text(normalize_entity_name(raw_name, entity_type)),
                normalize_text(normalize_entity_name(raw_name, entity_type, query_context=query_context)),
                *self._umls_expanded_names(raw_name, entity_type, query_context=query_context),
            ]
        )
        if not candidate_names:
            return None

        normalized = candidate_names[0]
        has_known_alias = any(name != raw_normalized for name in candidate_names)
        raw_tokens = raw_normalized.split()
        original_tokens = str(raw_name).strip().split()
        is_short_ambiguous = (
            raw_normalized in SHORT_AMBIGUOUS_ENTITY_NAMES
            or len(raw_normalized) <= 3
            or (
                len(raw_tokens) == 1
                and len(raw_tokens[0]) <= 4
                and bool(original_tokens)
                and original_tokens[0].isupper()
            )
        )

        candidates = []
        matched_query_name = normalized
        for candidate_name in candidate_names:
            exact_candidates = self.name_index.get(candidate_name, [])
            if exact_candidates:
                candidates = exact_candidates
                matched_query_name = candidate_name
                break

        if not candidates:
            if is_short_ambiguous and not has_known_alias:
                return None
            substring_candidates = []
            for candidate_name in candidate_names:
                if len(candidate_name) < 5:
                    continue
                for key, values in self.name_index.items():
                    if len(key) >= 5 and (candidate_name in key or key in candidate_name):
                        substring_candidates.extend(values)
                        if len(substring_candidates) >= 20:
                            break
                if len(substring_candidates) >= 20:
                    break
            candidates = substring_candidates[:20]
            if candidates:
                matched_query_name = max(
                    candidate_names,
                    key=lambda name: max(
                        SequenceMatcher(None, name, normalize_text(c["name"])).ratio()
                        for c in candidates
                    ),
                )

        if not candidates:
            fuzzy_names = [name for name in candidate_names if len(name) >= 5]
            if (is_short_ambiguous and not has_known_alias) or not fuzzy_names:
                return None
            best = None
            best_score = 0.0
            best_query_name = fuzzy_names[0]
            for key, values in self.name_index.items():
                for candidate_name in fuzzy_names:
                    score = SequenceMatcher(None, candidate_name, key).ratio()
                    if score > best_score:
                        best_score = score
                        best = values[0]
                        best_query_name = candidate_name
            if best is None or best_score < 0.92:
                return None
            candidates = [best]
            matched_query_name = best_query_name

        filtered = [c for c in candidates if c["entity_type"] == entity_type] or candidates

        umls_names = set(self._umls_expanded_names(raw_name, entity_type, query_context=query_context))

        def candidate_score(c: Dict[str, str]) -> float:
            name_norm = normalize_text(c["name"])
            candidate_best = max(
                SequenceMatcher(None, candidate_name, name_norm).ratio()
                for candidate_name in candidate_names
            )
            umls_bonus = 0.25 if name_norm in umls_names else 0.0
            exact_bonus = 0.35 if name_norm == raw_normalized else 0.0
            specificity_bonus = self._specificity_bonus(raw_name, c["name"])
            generic_penalty = self._generic_parent_penalty(raw_name, c["name"], entity_type)
            focus_bonus = self._question_match_bonus(c["name"], focus_keywords or [])
            gate_penalty = self._question_gate_penalty(raw_name, c["name"], focus_keywords or [])
            return (
                candidate_best
                + umls_bonus
                + exact_bonus
                + specificity_bonus
                + focus_bonus
                - generic_penalty
                - gate_penalty
            )

        best = max(
            filtered,
            key=candidate_score,
        )
        best_normalized = normalize_text(best["name"])
        score = max(SequenceMatcher(None, candidate_name, best_normalized).ratio() for candidate_name in candidate_names)
        matched_query_name = max(
            candidate_names,
            key=lambda candidate_name: SequenceMatcher(None, candidate_name, best_normalized).ratio(),
        )
        if best["entity_type"] != entity_type:
            return None
        if matched_query_name != best_normalized:
            if is_short_ambiguous and matched_query_name == raw_normalized:
                return None
            if len(matched_query_name) < 5 or len(best_normalized) < 5:
                return None
            if score < 0.90 and not (matched_query_name in best_normalized or best_normalized in matched_query_name):
                return None
        return PrimeKGEntity(
            raw_name=raw_name,
            normalized_name=matched_query_name,
            matched_name=best["name"],
            entity_type=entity_type,
            score=score,
        )

    def _question_entity_gate(
        self,
        entity: PrimeKGEntity,
        focus_keywords: Sequence[str],
    ) -> float:
        return max(
            self._focus_overlap_score(entity.raw_name, focus_keywords),
            self._focus_overlap_score(entity.normalized_name, focus_keywords),
            self._focus_overlap_score(entity.matched_name, focus_keywords),
        )

    def _format_relation_text(self, row: Dict[str, Any]) -> str:
        x_name = str(row.get("x_name", "")).strip()
        y_name = str(row.get("y_name", "")).strip()
        x_type = self._simplify_type(str(row.get("x_type", "")))
        y_type = self._simplify_type(str(row.get("y_type", "")))
        relation_key = self._relation_key(row)

        if relation_key == "contraindication":
            if x_type == "drug" and y_type != "drug":
                return f"PrimeKG suggests that {x_name} has a contraindication relationship with {y_name}."
            if y_type == "drug" and x_type != "drug":
                return f"PrimeKG suggests that {y_name} has a contraindication relationship with {x_name}."
            return f"PrimeKG suggests that {x_name} has a contraindication relationship with {y_name}."
        if relation_key == "indication":
            if x_type == "drug" and y_type != "drug":
                return f"PrimeKG suggests that {x_name} is indicated for {y_name}."
            if y_type == "drug" and x_type != "drug":
                return f"PrimeKG suggests that {y_name} is indicated for {x_name}."
            return f"PrimeKG suggests that {x_name} is indicated for {y_name}."
        if relation_key == "disease disease":
            return f"PrimeKG suggests that {x_name} is associated with {y_name}."
        if relation_key == "phenotype positive":
            return f"PrimeKG suggests that {y_name} is positively associated with {x_name}."
        if relation_key == "phenotype negative":
            return f"PrimeKG suggests that {y_name} is negatively associated with {x_name}."
        return f"PrimeKG relation: {x_name} | {row.get('display_relation', row.get('relation', 'related to'))} | {y_name}"

    def _relation_allowed(self, row: Dict[str, Any], question_type: str | None) -> bool:
        allowed = QUESTION_TYPE_RELATIONS.get(question_type or "yes", QUESTION_TYPE_RELATIONS["yes"])
        relation_key = self._relation_key(row)
        return relation_key in allowed or relation_key in RELATION_PRIORITY

    def _score_relation_with_question(
        self,
        row: Dict[str, Any],
        matched_entities: List[PrimeKGEntity],
        core_entities: List[PrimeKGEntity],
        focus_keywords: Sequence[str],
        question_type: str | None = None,
    ) -> float:
        score = 0.0
        x_name = normalize_text(row.get("x_name", ""))
        y_name = normalize_text(row.get("y_name", ""))
        relation_key = self._relation_key(row)
        row_text = " ".join(
            str(row.get(k, "")) for k in ("x_name", "y_name", "relation", "display_relation", "x_type", "y_type")
        )

        for entity in matched_entities:
            matched = normalize_text(entity.matched_name)
            if matched in {x_name, y_name}:
                score += 2.0
            elif entity.normalized_name in {x_name, y_name}:
                score += 1.5

        for entity in core_entities:
            matched = normalize_text(entity.matched_name)
            if matched in {x_name, y_name}:
                score += 4.0

        score += RELATION_PRIORITY.get(relation_key, 0.5)
        score += self._focus_overlap_score(row_text, focus_keywords)

        if question_type == "numeric" and relation_key not in QUESTION_TYPE_RELATIONS["numeric"]:
            score -= 2.0

        return score

    def _entity_row_ids(self, entity: PrimeKGEntity) -> set[int]:
        row_ids: set[int] = set()
        for key in {
            normalize_text(entity.matched_name),
            normalize_text(entity.normalized_name),
            normalize_text(entity.raw_name),
        }:
            row_ids.update(self.row_index.get(key, []))
        return row_ids

    def _row_endpoint_keys(self, row: Dict[str, Any]) -> set[str]:
        return {
            normalize_text(row.get("x_name", "")),
            normalize_text(row.get("y_name", "")),
        }

    def _shared_endpoint(self, first: Dict[str, Any], second: Dict[str, Any]) -> str:
        shared = self._row_endpoint_keys(first) & self._row_endpoint_keys(second)
        return next((x for x in shared if x), "")

    def _format_path_text(self, first: Dict[str, Any], second: Dict[str, Any]) -> str:
        shared = self._shared_endpoint(first, second)
        if not shared:
            return ""

        def row_text(row: Dict[str, Any]) -> str:
            relation = row.get("display_relation") or row.get("relation") or "related to"
            return f"{row.get('x_name', '')} --{relation}-- {row.get('y_name', '')}"

        return f"{row_text(first)}; {row_text(second)}"

    def _is_mirror_two_hop_path(self, first: Dict[str, Any], second: Dict[str, Any]) -> bool:
        first_x = normalize_text(first.get("x_name", ""))
        first_y = normalize_text(first.get("y_name", ""))
        second_x = normalize_text(second.get("x_name", ""))
        second_y = normalize_text(second.get("y_name", ""))
        if not all([first_x, first_y, second_x, second_y]):
            return False

        first_relation = normalize_text(first.get("relation", ""))
        second_relation = normalize_text(second.get("relation", ""))
        first_display = normalize_text(first.get("display_relation", ""))
        second_display = normalize_text(second.get("display_relation", ""))
        return (
            first_x == second_y
            and first_y == second_x
            and first_relation == second_relation
            and first_display == second_display
        )

    def _build_two_hop_paths(
        self,
        core_entities: List[PrimeKGEntity],
        support_entities: List[PrimeKGEntity],
        focus_keywords: Sequence[str],
        question_type: str | None = None,
        max_paths: int = 3,
    ) -> List[PrimeKGPath]:
        if not core_entities or max_paths <= 0:
            return []

        all_entities = core_entities + support_entities
        support_names = {normalize_text(e.matched_name) for e in support_entities}
        core_names = {normalize_text(e.matched_name) for e in core_entities}
        paths: List[PrimeKGPath] = []

        for entity in core_entities:
            first_rows = [
                self.rows[idx]
                for idx in self._entity_row_ids(entity)
                if self._relation_allowed(self.rows[idx], question_type)
            ]
            first_rows = sorted(
                first_rows,
                key=lambda row: self._score_relation_with_question(
                    row, all_entities, core_entities, focus_keywords, question_type
                ),
                reverse=True,
            )[:25]

            for first in first_rows:
                for shared_key in self._row_endpoint_keys(first):
                    if not shared_key or shared_key in core_names:
                        continue
                    second_ids = self.row_index.get(shared_key, [])[:80]
                    for second_idx in second_ids:
                        second = self.rows[second_idx]
                        if second is first or not self._relation_allowed(second, question_type):
                            continue
                        if not self._shared_endpoint(first, second):
                            continue
                        if self._is_mirror_two_hop_path(first, second):
                            continue

                        endpoint_score = 0.0
                        endpoints = self._row_endpoint_keys(second)
                        if endpoints & core_names:
                            endpoint_score += 2.0
                        if endpoints & support_names:
                            endpoint_score += 2.5

                        score = (
                            self._score_relation_with_question(
                                first, all_entities, core_entities, focus_keywords, question_type
                            )
                            + self._score_relation_with_question(
                                second, all_entities, core_entities, focus_keywords, question_type
                            )
                            + endpoint_score
                        )
                        text = self._format_path_text(first, second)
                        if not text:
                            continue
                        paths.append(PrimeKGPath(text=text, score=score, rows=[first, second]))

        paths.sort(key=lambda item: item.score, reverse=True)
        deduped: List[PrimeKGPath] = []
        seen = set()
        for path in paths:
            row_keys = [
                "|".join(
                    normalize_text(str(row.get(field, "")))
                    for field in ("x_name", "relation", "display_relation", "y_name")
                )
                for row in path.rows
            ]
            key = "||".join(sorted(row_keys))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(path)
            if len(deduped) >= max_paths:
                break
        return deduped

    def lookup_relations(
        self,
        entities: List[PrimeKGEntity],
        core_entities: List[PrimeKGEntity] | None = None,
        focus_keywords: Sequence[str] | None = None,
        question_type: str | None = None,
        max_relations: int = 10,
    ) -> List[PrimeKGRelation]:
        if not entities:
            return []

        core_entities = core_entities or []
        focus_keywords = focus_keywords or []
        candidate_source = core_entities or entities
        if core_entities:
            candidate_source = self._dedup_entities(
                list(candidate_source) + [entity for entity in entities if entity.entity_type == "drug"]
            )
        candidate_rows = set()
        for entity in candidate_source:
            candidate_rows.update(self._entity_row_ids(entity))

        relations: List[PrimeKGRelation] = []
        for idx in candidate_rows:
            row = self.rows[idx]
            if not self._relation_allowed(row, question_type):
                continue
            if not self._relation_is_query_specific(
                row=row,
                entities=entities,
                core_entities=core_entities or entities,
                focus_keywords=focus_keywords,
            ):
                continue

            score = self._score_relation_with_question(
                row=row,
                matched_entities=entities,
                core_entities=core_entities,
                focus_keywords=focus_keywords,
                question_type=question_type,
            )
            if score <= 0:
                continue

            relations.append(
                PrimeKGRelation(
                    text=self._format_relation_text(row),
                    score=score,
                    row=row,
                )
            )

        relations.sort(key=lambda item: item.score, reverse=True)
        deduped = []
        seen = set()
        for relation in relations:
            key = relation.text.lower().strip()
            if key not in seen:
                seen.add(key)
                deduped.append(relation)
            if len(deduped) >= max_relations:
                break
        return deduped

    def build_primekg_evidence_from_structured_entities(
        self,
        entities_dict: Dict[str, List[str]],
        question: str = "",
        sub_queries: Sequence[str] | None = None,
        max_relations: int = 6,
        max_paths: int = 3,
        question_type: str | None = None,
    ) -> Dict[str, Any]:
        cache_key = self._cache_key(entities_dict, question_type, question=question, sub_queries=sub_queries)
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        focus_keywords = self._question_focus_keywords(question, sub_queries)
        query_context = self._build_query_context(question, sub_queries)
        normalized = normalize_entities(
            entities_dict.get("diseases", []),
            entities_dict.get("drugs", []),
        )

        entities: List[PrimeKGEntity] = []
        for raw_name in normalized["diseases_raw"]:
            matched = self._match_entity(
                raw_name,
                "disease",
                focus_keywords=focus_keywords,
                query_context=query_context,
            )
            if matched is not None:
                entities.append(matched)
        for raw_name in normalized["drugs_raw"]:
            matched = self._match_entity(
                raw_name,
                "drug",
                focus_keywords=focus_keywords,
                query_context=query_context,
            )
            if matched is not None:
                entities.append(matched)

        focus_scored = [(entity, self._entity_focus_score(entity, focus_keywords)) for entity in entities]
        gated_focus_scored = [
            (entity, score, self._question_entity_gate(entity, focus_keywords))
            for entity, score in focus_scored
        ]
        core_entities = [
            entity
            for entity, score, gate in gated_focus_scored
            if score >= 2.0 and (gate > 0.0 or not focus_keywords)
        ]
        if not core_entities and gated_focus_scored and focus_keywords:
            gated_candidates = [item for item in gated_focus_scored if item[2] > 0.0]
            if gated_candidates:
                core_entities = [max(gated_candidates, key=lambda item: (item[2], item[1], item[0].score))[0]]
        if not core_entities and entities:
            core_entities = [max(entities, key=lambda entity: (self._question_entity_gate(entity, focus_keywords), self._entity_focus_score(entity, focus_keywords), entity.score))]
        support_entities = [entity for entity in entities if entity not in core_entities]

        relations = self.lookup_relations(
            entities=entities,
            core_entities=core_entities,
            focus_keywords=focus_keywords,
            question_type=question_type,
            max_relations=max_relations,
        )
        paths = self._build_two_hop_paths(
            core_entities=core_entities,
            support_entities=support_entities,
            focus_keywords=focus_keywords,
            question_type=question_type,
            max_paths=max_paths,
        )
        paths = [
            path
            for path in paths
            if self._path_is_query_specific(path, entities, focus_keywords)
        ]

        diseases = self._dedup_keep_order(
            [entity.matched_name for entity in entities if entity.entity_type == "disease"]
        )
        drugs = self._dedup_keep_order(
            [entity.matched_name for entity in entities if entity.entity_type == "drug"]
        )
        core_diseases = self._dedup_keep_order(
            [entity.matched_name for entity in core_entities if entity.entity_type == "disease"]
        )
        core_drugs = self._dedup_keep_order(
            [entity.matched_name for entity in core_entities if entity.entity_type == "drug"]
        )
        support_diseases = self._dedup_keep_order(
            [entity.matched_name for entity in support_entities if entity.entity_type == "disease"]
        )
        support_drugs = self._dedup_keep_order(
            [entity.matched_name for entity in support_entities if entity.entity_type == "drug"]
        )

        lines = []
        if core_diseases or core_drugs:
            lines.append("PrimeKG aligned core entities:")
            for disease in core_diseases:
                lines.append(f"- Disease: {disease}")
            for drug in core_drugs:
                lines.append(f"- Drug: {drug}")
        if support_diseases or support_drugs:
            lines.append("PrimeKG aligned support entities:")
            if support_diseases:
                lines.append("- Diseases: " + ", ".join(support_diseases))
            if support_drugs:
                lines.append("- Drugs: " + ", ".join(support_drugs))
        if relations:
            lines.append("PrimeKG high-value relations:")
            for relation in relations:
                lines.append(f"- {relation.text}")
        if paths:
            lines.append("PrimeKG short reasoning paths:")
            for path in paths:
                lines.append(f"- {path.text}")

        result = {
            "diseases": diseases,
            "drugs": drugs,
            "primekg_evidence": "\n".join(lines).strip(),
        }
        self._memory_cache[cache_key] = result
        return result