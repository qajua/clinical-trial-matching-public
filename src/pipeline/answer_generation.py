import re
import json


YESNO_SYSTEM_PROMPT = """
You are a clinical trial eligibility assistant.

You will be given:
1. The question
2. Decomposed sub-queries
3. Retrieved evidence chunks from a clinical note
4. Optional disease/drug hints
5. Optional PrimeKG supporting evidence

Rules:
- Use the question and sub-queries to understand the target condition or value
- Use the retrieved evidence chunks as the primary evidence
- Use disease/drug hints and PrimeKG evidence only as supporting context
- Decide the final answer first
- For yes/no questions, the answer must be exactly one of: Yes or No
- Only answer Yes when there is direct and explicit evidence in the note
- If the note is negative, absent, not mentioned, historical-only, planned-only, or too indirect, prefer No
- Do not infer Yes from loosely related disease, drug, or PrimeKG context alone
- Do not output chain-of-thought
- Return ONLY valid JSON with exactly these keys:
  {
    "answer": "Yes",
    "explanation": "short evidence-based explanation"
  }
- Put the final answer in "answer" first
- Keep "explanation" short and grounded in the note

Example:
{
  "answer": "Yes",
  "explanation": "The note explicitly documents atrial fibrillation."
}
"""


NUMERIC_SYSTEM_PROMPT = """
You are a clinical trial eligibility assistant.

You will be given:
1. The question
2. Decomposed sub-queries
3. Retrieved evidence chunks from a clinical note
4. Optional disease/drug hints
5. Optional PrimeKG supporting evidence

Rules:
- Use the question and sub-queries to understand the target value
- Use the retrieved evidence chunks as the primary evidence
- Use disease/drug hints and PrimeKG evidence only as supporting context
- Decide the final answer first
- For numeric questions, the answer must be either:
  - a bare numeric string such as 8.1, 100, 55
  - or NA
- Do not include units inside the answer field
- If the value is not supported by the note, answer NA
- Do not output chain-of-thought
- Return ONLY valid JSON with exactly these keys:
  {
    "answer": "8.1",
    "explanation": "short evidence-based explanation"
  }
- Put the final answer in "answer" first
- Keep "explanation" short and grounded in the note

Example:
{
  "answer": "70",
  "explanation": "The note reports left ventricular ejection fraction of 70%."
}
"""


def build_prompt(
    question: str,
    sub_queries: list[str],
    top_chunks: list[str],
    facts: dict,
    primekg_evidence: str,
) -> str:
    question_text = question.strip() if str(question).strip() else "None"
    sub_queries_text = "\n".join(sub_queries) if sub_queries else "None"
    chunks_text = "\n\n---\n\n".join(top_chunks) if top_chunks else "None"

    facts_lines = []
    if facts:
        if facts.get("diseases"):
            facts_lines.append("Diseases: " + ", ".join(facts["diseases"]))
        if facts.get("drugs"):
            facts_lines.append("Drugs: " + ", ".join(facts["drugs"]))
    facts_text = "\n".join(facts_lines) if facts_lines else "None"
    primekg_text = primekg_evidence if primekg_evidence.strip() else "None"

    return f"""Question:
{question_text}

Decomposed sub-queries:
{sub_queries_text}

Retrieved evidence chunks:
{chunks_text}

Disease/drug hints:
{facts_text}

PrimeKG evidence:
{primekg_text}
"""


class AnswerGenerator:
    def __init__(self, llm_client, gen_cfg: dict, dataset_cfg: dict):
        self.llm = llm_client
        self.temperature = gen_cfg.get("temperature", 0.1)
        self.max_tokens = dataset_cfg.get("answer_max_tokens", 50)

    @staticmethod
    def _extract_json_field(prediction: str, field: str) -> str | None:
        text = str(prediction).strip()
        if not text:
            return None

        regex_patterns = [
            rf'"{re.escape(field)}"\s*:\s*"([^"]*)"',
            rf"'{re.escape(field)}'\s*:\s*'([^']*)'",
            rf'"{re.escape(field)}"\s*:\s*([A-Za-z0-9.+-]+)',
            rf"'{re.escape(field)}'\s*:\s*([A-Za-z0-9.+-]+)",
        ]
        for pattern in regex_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                value = str(match.group(1)).strip()
                if value:
                    return value

        candidates = [text]
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            candidates.append(match.group(0))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                value = str(parsed.get(field, "")).strip()
                if value:
                    return value
        return None

    @classmethod
    def _postprocess_yesno(cls, prediction: str) -> str:
        json_answer = cls._extract_json_field(prediction, "answer")
        if json_answer:
            lowered = json_answer.strip().lower()
            if lowered in {"yes", "no"}:
                return lowered.capitalize()
            if lowered == "na":
                return "No"

        text = str(prediction).strip()
        if not text:
            return "No"

        first_line = text.splitlines()[0].strip().lower()
        if first_line in {"yes", "no"}:
            return first_line.capitalize()
        if first_line == "na":
            return "No"

        lowered_full = text.lower()
        if re.search(r"\byes\b", lowered_full):
            return "Yes"
        if re.search(r"\bno\b", lowered_full):
            return "No"
        return "No"

    @classmethod
    def _postprocess_numeric(cls, prediction: str) -> str:
        json_answer = cls._extract_json_field(prediction, "answer")
        if json_answer:
            cleaned_json_answer = str(json_answer).strip()
            lowered_json = cleaned_json_answer.lower()
            if lowered_json == "na" or "not stated" in lowered_json or "not reported" in lowered_json:
                return "NA"
            if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned_json_answer):
                return cleaned_json_answer

        text = str(prediction).strip()
        if not text:
            return "NA"

        lowered = text.lower()
        if lowered == "na" or "not stated" in lowered or "not reported" in lowered:
            return "NA"

        cleaned_lines = []
        for line in text.splitlines():
            line = re.sub(r"^\s*[-*]\s*", "", line)
            line = re.sub(r"^\s*\d+[\.\)]\s+", "", line)
            cleaned_lines.append(line)
        cleaned = "\n".join(cleaned_lines).strip()
        lowered_cleaned = cleaned.lower()
        if lowered_cleaned == "na" or "not stated" in lowered_cleaned or "not reported" in lowered_cleaned:
            return "NA"

        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned):
            return cleaned

        matches = re.findall(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?", cleaned)
        if not matches:
            return "NA"
        return matches[-1]

    def generate(
        self,
        qtype: str,
        question: str,
        sub_queries: list[str],
        top_chunks: list[str],
        facts: dict,
        primekg_evidence: str,
    ) -> str:
        system_prompt = YESNO_SYSTEM_PROMPT if qtype.lower() == "yes" else NUMERIC_SYSTEM_PROMPT
        user_prompt = build_prompt(
            question=question,
            sub_queries=sub_queries,
            top_chunks=top_chunks,
            facts=facts,
            primekg_evidence=primekg_evidence,
        )
        prediction = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        ).strip()

        if qtype.lower() == "yes":
            return self._postprocess_yesno(prediction)
        return self._postprocess_numeric(prediction)

    def generate_with_raw(
        self,
        qtype: str,
        question: str,
        sub_queries: list[str],
        top_chunks: list[str],
        facts: dict,
        primekg_evidence: str,
    ) -> tuple[str, str]:
        system_prompt = YESNO_SYSTEM_PROMPT if qtype.lower() == "yes" else NUMERIC_SYSTEM_PROMPT
        user_prompt = build_prompt(
            question=question,
            sub_queries=sub_queries,
            top_chunks=top_chunks,
            facts=facts,
            primekg_evidence=primekg_evidence,
        )
        raw_prediction = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        ).strip()
        if qtype.lower() == "yes":
            return raw_prediction, self._postprocess_yesno(raw_prediction)
        return raw_prediction, self._postprocess_numeric(raw_prediction)