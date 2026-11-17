#!/usr/bin/env python3
"""
Fixed LoCoMo runner (v2):
Fixes for the original run_locomo_bounded.py:
  1. Removes the 64-token answer cap → answers are no longer truncated mid-reasoning
  2. Replaces the 7-step scanning prompt with a concise direct prompt + thinking=OFF
     The 7-step prompt forces ~500 tokens of visible reasoning per question; on DGX
     (TP=1) this takes ~100s/question (7+ hours total).  With the simple prompt and
     thinking disabled the model gives the final answer directly in 2–5s on DGX and
     <1s on 5880ada (TP=4).  The skill notes confirm equal or better accuracy.
  3. Judge left completely unchanged — it already uses json_object mode; adding
     extra_body breaks json_object on DGX's SGLang build.
"""
import runpy

import benchmarks.locomo.prompts as _prompts
from benchmarks.common.llm_client import LLMClient

# ---------------------------------------------------------------------------
# Simple answerer prompt (replaces the 7-step scanning template)
# ---------------------------------------------------------------------------
SIMPLE_SYSTEM = (
    "You answer questions using retrieved memory records from past conversations. "
    "Be concise and give only the specific answer requested. "
    "Do not explain reasoning. Never say 'not found' — if any memory is relevant, use it."
)

def _simple_answer_prompt(question: str, search_results: list, reference_date: str = None,
                           user_profile: str = None) -> str:
    """Build a short, direct prompt from retrieved memories."""
    lines = []
    if reference_date:
        lines.append(f"These conversations occurred around {reference_date} (all events 2022-2024).")
    lines.append("Memory records (most relevant first):")
    for i, r in enumerate(search_results, 1):
        mem = r.get("memory") or r.get("content") or str(r)
        lines.append(f"  {i}. {mem}")
    lines.append(f"\nQuestion: {question}")
    lines.append("Answer:")
    return "\n".join(lines)

# Replace the prompt builder used by the runner
_prompts.get_answer_generation_prompt = _simple_answer_prompt

# ---------------------------------------------------------------------------
# Monkey-patch _generate_openai: thinking=OFF, 128-token cap (ample for answers)
# ---------------------------------------------------------------------------
_orig_generate_openai = LLMClient._generate_openai
THINKING_OFF = {"chat_template_kwargs": {"enable_thinking": False}}
ANSWER_MAX_TOKENS = 128   # factual answers rarely exceed 60 tokens; 128 is safe


async def _fixed_generate_openai(self, system: str, user: str,
                                  temperature: float, max_tokens: int) -> str:
    """Answer generation: thinking=OFF, 128-token cap."""
    import asyncio as _asyncio
    capped = min(max_tokens, ANSWER_MAX_TOKENS)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    for attempt in range(self.max_retries):
        try:
            async with self.limiter:
                resp = await _asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        extra_body=THINKING_OFF,
                        **self._openai_chat_temperature_kwargs(temperature),
                        **self._openai_chat_token_limit_kwargs(capped),
                    ),
                    timeout=self.timeout,
                )
            content = resp.choices[0].message.content
            if content is None:
                content = ""
            return content.strip()
        except Exception:
            if attempt < self.max_retries - 1:
                await _asyncio.sleep(2 ** attempt)
    return ""


LLMClient._generate_openai = _fixed_generate_openai

# _generate_structured_openai (judge) is intentionally left unchanged:
#   - uses response_format={"type":"json_object"} → valid JSON
#   - adding extra_body breaks json_object on DGX's SGLang build
#   - 4096-token cap is fine; the judge prompt + JSON output is <200 tokens

runpy.run_module("benchmarks.locomo.run", run_name="__main__")
