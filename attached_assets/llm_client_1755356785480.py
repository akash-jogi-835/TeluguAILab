# llm_client.py
import os
from typing import Optional, List, Dict

# --- OpenAI client (preferred) ---
_OPENAI_AVAILABLE = True
try:
    from openai import OpenAI  # pip install openai>=1.40
except Exception:
    _OPENAI_AVAILABLE = False

# --- Optional local fallback via transformers (nice-to-have) ---
_TRANSFORMERS_AVAILABLE = True
try:
    from transformers import pipeline  # pip install transformers
except Exception:
    _TRANSFORMERS_AVAILABLE = False


class LLMClient:
    """
    Small wrapper that supports:
      - OpenAI (default)
      - Optional local Transformers fallback if OpenAI isn't configured
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",     # fast + cheap, change if you want
        temperature: float = 0.2,
        max_tokens: int = 400,
        openai_api_key: Optional[str] = None,
    ):
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if self.provider == "openai":
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
            self._setup_openai()

        elif self.provider == "local":
            self._setup_local()

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    # ---------- Provider setups ----------
    def _setup_openai(self):
        if not _OPENAI_AVAILABLE:
            raise RuntimeError("openai package not installed. Run: uv add openai")
        # Will read OPENAI_API_KEY from env if not passed explicitly
        self._client = OpenAI()

    def _setup_local(self):
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers not installed. Run: uv add transformers"
            )
        # You can change this to any small local model available to you
        # (for CPU-only, something like 'google/flan-t5-base' works for summarize/QA)
        self._summarizer = pipeline("summarization", model="google/flan-t5-base")
        self._qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

    # ---------- High-level helpers ----------
    def summarize(self, text: str, max_chars: int = 700) -> str:
        prompt = (
            "You are a concise multilingual assistant (English + Telugu). "
            "Summarize the text below into a clear paragraph, preserving key facts "
            "and named entities. Keep it under about 5 sentences.\n\n"
            f"Text:\n{text}"
        )
        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": "You write concise, factual summaries."},
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content.strip()

        # local transformers
        out = self._summarizer(text[:3000], max_length=180, min_length=60, do_sample=False)
        return out[0]["summary_text"]

    def answer(self, context: str, question: str) -> str:
        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": "Answer strictly using the given context. If missing, say you don't know."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
            )
            return resp.choices[0].message.content.strip()

        # local transformers
        out = self._qa(question=question, context=context[:4000])
        if out.get("answer"):
            return out["answer"]
        return "I couldn't find the answer in the text."

    def key_phrases(self, text: str) -> List[str]:
        # Light LLM prompt (works with OpenAI); for local, just return top tokens
        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=200,
                messages=[
                    {"role": "system", "content": "Extract up to 10 key phrases (comma-separated)."},
                    {"role": "user", "content": text[:4000]},
                ],
            )
            line = resp.choices[0].message.content.strip()
            parts = [p.strip() for p in line.split(",") if p.strip()]
            return parts[:10]

        # local heuristic
        import re
        words = re.findall(r"\b\w{4,}\b", text.lower())
        freq: Dict[str, int] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]]
