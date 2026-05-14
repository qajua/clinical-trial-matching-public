import csv
import json
import os
import re
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = Path(__file__).resolve().parent

DEMO_NOTES_PATH = Path(
    os.environ.get("DEMO_NOTES_PATH", ROOT / "data" / "demo" / "demo_notes.csv")
)

DEMO_PRIMEKG_PATH = Path(
    os.environ.get("DEMO_PRIMEKG_PATH", ROOT / "data" / "demo" / "demo_primekg.csv")
)


def read_csv_rows(path: Path):
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def clean_text(value):
    return " ".join(str(value or "").split())


def tokenize(text):
    stopwords = {
        "the", "and", "or", "of", "to", "in", "on", "for", "with", "without",
        "does", "note", "patient", "describe", "having", "have", "has", "had",
        "what", "is", "are", "was", "were", "answer", "mentioned", "highest",
        "lowest", "available", "ever", "this", "that", "from", "into", "a", "an",
    }
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9]+", str(text).lower())
        if len(token) >= 3 and token not in stopwords
    ]


def infer_question_type(question):
    q = str(question or "").strip().lower()

    numeric_markers = [
        "what is the highest",
        "what is the lowest",
        "how many",
        "value",
        "level",
        "creatinine",
        "hemoglobin",
        "platelet",
        "age",
        "bmi",
    ]

    if any(marker in q for marker in numeric_markers):
        return "numeric"

    return "yes"


def build_query_optimization(question):
    q = clean_text(question)
    q_lower = q.lower()

    rewritten = q

    if "heart failure" in q_lower:
        rewritten = "evidence of heart failure diagnosis, congestive heart failure, or reduced ejection fraction"
        sub_queries = [
            "heart failure diagnosis",
            "congestive heart failure history",
            "reduced ejection fraction or cardiac dysfunction",
        ]
    elif "hypertension" in q_lower or "arterial hypertension" in q_lower:
        rewritten = "evidence of hypertension diagnosis or antihypertensive treatment"
        sub_queries = [
            "hypertension diagnosis",
            "arterial hypertension on treatment",
            "blood pressure medication evidence",
        ]
    elif "stroke" in q_lower or "transient ischemic attack" in q_lower or "tia" in q_lower:
        rewritten = "evidence of prior stroke or transient ischemic attack"
        sub_queries = [
            "history of stroke",
            "transient ischemic attack",
            "cerebrovascular event history",
        ]
    elif "creatinine" in q_lower:
        rewritten = "serum creatinine laboratory value mentioned in the note"
        sub_queries = [
            "serum creatinine value",
            "highest creatinine",
            "renal function laboratory result",
        ]
    elif "hemoglobin" in q_lower:
        rewritten = "hemoglobin laboratory value mentioned in the note"
        sub_queries = [
            "hemoglobin value",
            "lowest hemoglobin",
            "blood count laboratory result",
        ]
    else:
        rewritten = q
        sub_queries = [q]

    return {
        "source": "rule_based_public_demo",
        "rewritten_query": rewritten,
        "sub_queries": sub_queries,
        "retrieval_query": rewritten,
    }


def split_note_into_chunks(note_text, max_chars=420):
    text = clean_text(note_text)
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    if not chunks:
        chunks = [text[:max_chars]]

    return chunks


def retrieve_top_chunks(note_text, retrieval_query, top_k=3):
    chunks = split_note_into_chunks(note_text)
    query_terms = tokenize(retrieval_query)

    scored = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = sum(1 for term in query_terms if term in chunk_lower)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected = [chunk for score, chunk in scored[:top_k] if chunk.strip()]

    if not selected and chunks:
        selected = chunks[:top_k]

    return selected


def extract_entities(question, chunks):
    text = f"{question} " + " ".join(chunks)
    text_lower = text.lower()

    disease_aliases = {
        "heart failure": ["heart failure", "congestive heart failure", "reduced ejection fraction"],
        "hypertension": ["hypertension", "arterial hypertension", "high blood pressure"],
        "stroke": ["stroke", "transient ischemic attack", "tia"],
        "renal impairment": ["creatinine", "renal", "kidney"],
        "anemia": ["hemoglobin", "anaemia", "anemia"],
    }

    drug_aliases = {
        "apixaban": ["apixaban", "eliquis"],
        "warfarin": ["warfarin"],
        "antihypertensive therapy": ["antihypertensive", "blood pressure medication"],
    }

    diseases = []
    drugs = []

    for canonical, aliases in disease_aliases.items():
        if any(alias in text_lower for alias in aliases):
            diseases.append(canonical)

    for canonical, aliases in drug_aliases.items():
        if any(alias in text_lower for alias in aliases):
            drugs.append(canonical)

    return {
        "diseases": diseases,
        "drugs": drugs,
        "primekg_diseases": diseases,
        "primekg_drugs": drugs,
    }


def build_primekg_evidence(question, diseases, drugs):
    rows = read_csv_rows(DEMO_PRIMEKG_PATH)

    if not rows:
        return {
            "diseases": diseases,
            "drugs": drugs,
            "primekg_evidence": "",
        }

    wanted_entities = {x.lower() for x in diseases + drugs}
    evidence_items = []

    for row in rows:
        source = clean_text(row.get("source_name", ""))
        target = clean_text(row.get("target_name", ""))
        relation = clean_text(row.get("relation", ""))
        evidence = clean_text(row.get("evidence", ""))

        source_l = source.lower()
        target_l = target.lower()

        if source_l in wanted_entities or target_l in wanted_entities:
            if evidence:
                evidence_items.append(evidence)
            elif source and relation and target:
                evidence_items.append(f"{source} -- {relation} -- {target}")

    evidence_items = evidence_items[:5]

    return {
        "diseases": diseases,
        "drugs": drugs,
        "primekg_evidence": " ".join(evidence_items),
    }


def extract_numeric_answer(question, chunks):
    q = question.lower()
    text = " ".join(chunks)

    numbers = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)", text):
        try:
            numbers.append(float(match.group(1)))
        except ValueError:
            pass

    if not numbers:
        return None

    if "lowest" in q:
        return str(min(numbers))

    if "highest" in q:
        return str(max(numbers))

    return str(numbers[0])


def rule_based_boolean_answer(question, chunks):
    q = question.lower()
    text = " ".join(chunks).lower()

    positive_terms = []

    if "heart failure" in q:
        positive_terms = ["heart failure", "congestive heart failure", "reduced ejection fraction"]
    elif "hypertension" in q:
        positive_terms = ["hypertension", "arterial hypertension", "high blood pressure"]
    elif "stroke" in q or "transient ischemic attack" in q or "tia" in q:
        positive_terms = ["stroke", "transient ischemic attack", "tia"]
    else:
        question_terms = tokenize(question)
        positive_terms = question_terms[:5]

    negation_markers = [
        "no documented history",
        "no history",
        "denies",
        "without evidence",
        "not documented",
    ]

    if any(marker in text for marker in negation_markers) and any(term in text for term in positive_terms):
        return "No"

    if any(term in text for term in positive_terms):
        return "Yes"

    return "No"


def generate_with_llm(question, question_type, top_chunks, primekg_evidence):
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    if not api_key:
        return "", None

    try:
        import requests

        system_prompt = (
            "You are a clinical trial eligibility matching assistant. "
            "Answer strictly based on the provided demo evidence. "
            "For Boolean questions, answer only Yes or No. "
            "For numeric questions, answer only the numeric value."
        )

        user_prompt = f"""
Question type: {question_type}

Question:
{question}

Retrieved evidence:
{chr(10).join(f"- {chunk}" for chunk in top_chunks)}

PrimeKG evidence:
{primekg_evidence or "N/A"}

Return only the final answer.
""".strip()

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 64,
        }

        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        raw = data["choices"][0]["message"]["content"].strip()
        answer = raw.strip().splitlines()[0].strip()
        return raw, answer

    except Exception:
        return "", None


class MatchService:
    def __init__(self):
        self.notes_df = read_csv_rows(DEMO_NOTES_PATH)

    def notes(self, limit=120):
        rows = []

        for row in self.notes_df[:limit]:
            text = clean_text(row.get("text", ""))
            rows.append(
                {
                    "note_id": str(row.get("note_id", "")),
                    "hadm_id": str(row.get("hadm_id", "")),
                    "preview": text[:180],
                }
            )

        return rows

    def _select_note(self, note_id):
        if note_id:
            for row in self.notes_df:
                if str(row.get("note_id", "")) == str(note_id):
                    return row

            raise ValueError(f"Unknown note_id: {note_id}")

        if not self.notes_df:
            raise ValueError("No demo notes found. Please check data/demo/demo_notes.csv.")

        return self.notes_df[0]

    def match(self, question, note_id=None, include_primekg=True):
        question = clean_text(question)

        if not question:
            raise ValueError("Question is required.")

        row = self._select_note(note_id)
        note_text = clean_text(row.get("text", ""))

        question_type = infer_question_type(question)
        qopt = build_query_optimization(question)

        top_chunks = retrieve_top_chunks(
            note_text=note_text,
            retrieval_query=qopt["retrieval_query"],
            top_k=3,
        )

        facts = extract_entities(question, top_chunks)

        kg_result = {
            "diseases": facts.get("primekg_diseases", []),
            "drugs": facts.get("primekg_drugs", []),
            "primekg_evidence": "",
        }

        if include_primekg:
            kg_result = build_primekg_evidence(
                question=question,
                diseases=facts.get("primekg_diseases", []),
                drugs=facts.get("primekg_drugs", []),
            )

        raw_prediction = ""
        answer_source = "rule_based_public_demo"

        if question_type == "numeric":
            answer = extract_numeric_answer(question, top_chunks)
        else:
            answer = None

        if answer is None:
            raw_prediction, llm_answer = generate_with_llm(
                question=question,
                question_type=question_type,
                top_chunks=top_chunks,
                primekg_evidence=kg_result.get("primekg_evidence", ""),
            )

            if llm_answer:
                answer = llm_answer
                answer_source = "llm_api"
            else:
                if question_type == "numeric":
                    answer = "Not found"
                else:
                    answer = rule_based_boolean_answer(question, top_chunks)

        return {
            "note": {
                "note_id": str(row.get("note_id", "")),
                "hadm_id": str(row.get("hadm_id", "")),
                "preview": note_text[:220],
                "match_score": None,
            },
            "question": question,
            "question_type": question_type,
            "medical_question": question,
            "query_optimization": qopt,
            "top_chunks": [
                {
                    "rank": index + 1,
                    "text": chunk,
                }
                for index, chunk in enumerate(top_chunks)
            ],
            "primekg": {
                "diseases": kg_result.get("diseases", []),
                "drugs": kg_result.get("drugs", []),
                "evidence": kg_result.get("primekg_evidence", ""),
                "error": None,
            },
            "answer": answer,
            "raw_prediction": raw_prediction or "Answered by public demo fallback logic.",
            "answer_source": answer_source,
        }


SERVICE = MatchService()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def _cors_headers(self):
        self.send_header(
            "Access-Control-Allow-Origin",
            os.environ.get("CORS_ALLOW_ORIGIN", "*"),
        )
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "service": "clinical-trial-matching-public-demo",
                },
            )
            return

        if parsed.path == "/api/notes":
            self._send_json(200, {"notes": SERVICE.notes()})
            return

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path != "/api/match":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(raw_body or "{}")

            result = SERVICE.match(
                question=payload.get("question", ""),
                note_id=payload.get("note_id"),
                include_primekg=bool(payload.get("include_primekg", True)),
            )

            self._send_json(200, result)

        except Exception as exc:
            traceback.print_exc()
            self._send_json(
                500,
                {
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=4),
                },
            )


def main():
    port = int(os.environ.get("PORT", os.environ.get("DASHBOARD_PORT", "8766")))
    host = os.environ.get("HOST", "0.0.0.0")

    server = ThreadingHTTPServer((host, port), Handler)

    print(f"Serving public clinical-trial matching demo on http://127.0.0.1:{port}/")
    print(f"Demo notes path: {DEMO_NOTES_PATH}")
    print(f"Demo PrimeKG path: {DEMO_PRIMEKG_PATH}")

    server.serve_forever()


if __name__ == "__main__":
    main()