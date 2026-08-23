"""Manual API smoke test.

This module is intentionally inert during pytest collection. Run it directly
after setting ZHIPU_API_KEY if you want to check the GLM endpoint.
"""

from __future__ import annotations

import os
import random
import time

import pytest
from openai import OpenAI


pytestmark = pytest.mark.manual


def call_with_retry(client: OpenAI, messages: list[dict[str, str]], max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="glm-4.7-flash",
                messages=messages,
                max_tokens=2048,
            )
        except Exception as exc:
            if "1302" not in str(exc) and "rate limit" not in str(exc).lower():
                raise
            wait_time = (2**attempt) + random.uniform(0, 1)
            print(f"Rate limited; retrying in {wait_time:.1f}s")
            time.sleep(wait_time)
    raise RuntimeError("Exceeded maximum retry attempts")


def main() -> None:
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        raise RuntimeError("Set ZHIPU_API_KEY before running this smoke test")

    client = OpenAI(
        api_key=api_key,
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )
    prompts = [
        "Say OK and return a short confidence tag.",
    ]
    for prompt in prompts:
        result = call_with_retry(client, [{"role": "user", "content": prompt}])
        print(result.choices[0].message.content)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
