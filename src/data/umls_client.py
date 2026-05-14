import json
import os
import re
import sqlite3
from functools import lru_cache
from typing import Any, Dict, List, Optional

import requests


def _normalize_umls_text(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    return re.sub(r"\s+", " ", normalized).strip()


class UMLSClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://uts-ws.nlm.nih.gov/rest",
        cache_path: Optional[str] = None,
    ) -> None:
        self.api_key = (api_key or os.getenv("UMLS_API_KEY") or "").strip()
        self.base_url = base_url.rstrip("/")
        self.cache_path = cache_path or os.getenv(
            "UMLS_CACHE_PATH",
            "outputs/umls_rest_cache.sqlite3",
        )
        self._session: requests.Session | None = None
        self._conn: sqlite3.Connection | None = None

        if self.api_key:
            self._session = requests.Session()
            self._init_cache()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self._session)

    def _init_cache(self) -> None:
        if not self.cache_path:
            return
        cache_dir = os.path.dirname(self.cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self._conn = sqlite3.connect(self.cache_path, timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS umls_search_cache (
                cache_key TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def _load_cache(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        if self._conn is None:
            return None
        row = self._conn.execute(
            "SELECT payload_json FROM umls_search_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row[0])
        except Exception:
            return None
        return payload if isinstance(payload, list) else None

    def _save_cache(self, cache_key: str, payload: List[Dict[str, Any]]) -> None:
        if self._conn is None:
            return
        self._conn.execute(
            """
            INSERT INTO umls_search_cache (cache_key, payload_json)
            VALUES (?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET payload_json = excluded.payload_json
            """,
            (cache_key, json.dumps(payload, ensure_ascii=False)),
        )
        self._conn.commit()

    def _request(self, path: str, **params: Any) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        assert self._session is not None
        params["apiKey"] = self.api_key
        response = self._session.get(
            f"{self.base_url}/{path.lstrip('/')}",
            params=params,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    def _fetch_atoms(self, cui: str, page_size: int = 25) -> List[str]:
        if not cui:
            return []
        try:
            data = self._request(f"content/current/CUI/{cui}/atoms", pageSize=page_size)
            atoms = data.get("result", [])
        except Exception:
            return []

        names: List[str] = []
        seen = set()
        for atom in atoms if isinstance(atoms, list) else []:
            name = str(atom.get("name", "")).strip()
            key = _normalize_umls_text(name)
            if key and key not in seen:
                seen.add(key)
                names.append(name)
        return names

    def search_candidates(self, entity_name: str) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        normalized = _normalize_umls_text(entity_name)
        if not normalized:
            return []

        cache_key = f"search::{normalized}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        results: List[Dict[str, Any]] = []
        for search_type in ("exact", "words"):
            try:
                data = self._request(
                    "search/current",
                    string=entity_name,
                    searchType=search_type,
                    pageSize=5,
                )
                rows = data.get("result", {}).get("results", [])
            except Exception:
                rows = []

            for row in rows if isinstance(rows, list) else []:
                cui = str(row.get("ui", "")).strip()
                name = str(row.get("name", "")).strip()
                if not cui or cui == "NONE" or not name:
                    continue
                synonyms = self._fetch_atoms(cui)
                results.append(
                    {
                        "name": name,
                        "cui": cui,
                        "synonyms": synonyms,
                        "source": "umls_api",
                    }
                )
            if results:
                break

        deduped: List[Dict[str, Any]] = []
        seen = set()
        for item in results:
            key = (str(item.get("cui", "")).strip(), _normalize_umls_text(item.get("name", "")))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        self._save_cache(cache_key, deduped)
        return deduped

    def best_match(
        self,
        entity_name: str,
        entity_type: str = "",
        query_context: str = "",
    ) -> Optional[Dict[str, Any]]:
        candidates = self.search_candidates(entity_name)
        if not candidates:
            return None

        entity_norm = _normalize_umls_text(entity_name)
        query_tokens = set(_normalize_umls_text(query_context).split())

        def score(candidate: Dict[str, Any]) -> tuple[float, int, int]:
            canonical = _normalize_umls_text(candidate.get("name", ""))
            synonyms = [
                _normalize_umls_text(x)
                for x in candidate.get("synonyms", []) or []
                if _normalize_umls_text(x)
            ]
            score_value = 0.0
            if canonical == entity_norm:
                score_value += 6.0
            elif entity_norm in synonyms:
                score_value += 5.0
            elif canonical and (entity_norm in canonical or canonical in entity_norm):
                score_value += 2.5
            elif any(entity_norm and (entity_norm in syn or syn in entity_norm) for syn in synonyms[:10]):
                score_value += 2.0

            best_overlap = 0
            for text in [canonical] + synonyms[:10]:
                best_overlap = max(best_overlap, len(query_tokens & set(text.split())))
            score_value += min(1.5, 0.3 * best_overlap)
            exact_synonym = int(entity_norm in synonyms)
            has_synonyms = int(bool(synonyms))
            return (score_value, exact_synonym, has_synonyms)

        best = max(candidates, key=score)
        return {
            "name": str(best.get("name", "")).strip(),
            "cui": str(best.get("cui", "")).strip(),
            "entity_type": str(entity_type or "").strip(),
            "synonyms": list(best.get("synonyms", []) or []),
            "source": "umls_api",
        }


@lru_cache(maxsize=1)
def get_umls_client() -> UMLSClient:
    return UMLSClient()