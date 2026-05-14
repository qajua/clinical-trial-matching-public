import os
import requests


class VLLMClient:
    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.base_url = (
            base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        response_format: dict | None = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format is not None:
            payload["response_format"] = response_format

        resp = requests.post(url, headers=headers, json=payload, timeout=300)

        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            message = f"{exc}; response body: {resp.text[:1000]}"
            raise requests.HTTPError(message, response=resp) from exc

        data = resp.json()
        return data["choices"][0]["message"]["content"]