# -*- coding: utf-8 -*-
"""
Hi there !

Created on Sat Dec  6 09:28:26 2025

@author: Jinyi

Have a nice day !
"""

# agent_module.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import json
import logging
from dataclasses import dataclass
from typing import Dict, Any

from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError

# ============ Logging ============
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ============ Data Structure ============
@dataclass
class SummaryResult:
    summary: str
    innovation: str
    results: str

    def to_dict(self):
        return {
            "summary": self.summary,
            "innovation": self.innovation,
            "results": self.results,
        }

# ============ Init OpenAI Client ============
def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or "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"

    if not api_key or "YOUR_API_KEY" in api_key:
        raise ValueError("❌ 你还没有设置 OpenAI API Key！")

    return OpenAI(api_key=api_key)

MODEL_NAME = "gpt-4.1-mini"

SYSTEM_PROMPT = """
You are an academic assistant.
Extract 3 things from a paper abstract:
1. Overall summary
2. Key innovations
3. Main experimental results
Output strictly in JSON with keys: summary, innovation, results.
"""

def _build_prompt(abstract: str) -> str:
    return f"""
Here is the abstract:

\"\"\"{abstract}\"\"\"

Please extract:
- A summary
- Innovations
- Main experimental results
"""

# ============ Main Function ============
def generate_summary(abstract: str) -> Dict[str, str]:
    if not abstract.strip():
        return SummaryResult(
            "No abstract provided", "N/A", "N/A"
        ).to_dict()

    client = _get_client()
    user_prompt = _build_prompt(abstract)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        data = json.loads(response.choices[0].message.content)

        return SummaryResult(
            data.get("summary", ""),
            data.get("innovation", ""),
            data.get("results", "")
        ).to_dict()

    except (APIConnectionError, RateLimitError, APIStatusError) as e:
        logger.error(f"LLM API error: {e}")
        return SummaryResult(
            summary=f"LLM error: {e}",
            innovation="N/A",
            results="N/A"
        ).to_dict()

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return SummaryResult(
            summary="Unexpected failure.",
            innovation="N/A",
            results="N/A"
        ).to_dict()


if __name__ == "__main__":
    test = "This paper proposes a new convolutional neural network..."
    print(generate_summary(test))
