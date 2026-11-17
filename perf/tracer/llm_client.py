"""
llm_client.py — 流式 LLM 客户端，测量 TTFT + token 计数
=========================================================
通过 OpenAI-compatible API（SGLang）发送 streaming 请求，
记录：
  - TTFT：从发送请求到收到第一个有效 token 的时间
  - Decode：从第一个 token 到最后一个 token 的时间
  - token 计数：来自 SGLang streaming usage（如果支持）

thinking 模式处理：
  Qwen3 的 </think>...  块会被过滤，
  TTFT 只在第一个非 thinking token 到达时触发。
"""

import time
import json
import requests
from dataclasses import dataclass

from .timer import PhasedTimer


@dataclass
class LlmResult:
    text: str              # 最终回答文本（已过滤 thinking 块）
    ttft_ms: float         # 从请求到第一个有效 token
    total_ms: float        # 从请求到最后一个 token
    decode_ms: float       # total_ms - ttft_ms
    prompt_tokens: int | None
    completion_tokens: int | None


def _parse_sse_stream(resp: requests.Response) -> tuple[list[str], float | None, float | None, dict]:
    """
    解析 SSE 流，过滤 thinking 块，返回：
      (chunks, ttft_perf, end_perf, usage_dict)
    """
    chunks: list[str] = []
    ttft_perf = None
    end_perf = None
    usage: dict = {}
    in_think = False

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str.strip() == "[DONE]":
            break

        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        # 提取 content delta
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                # ── 过滤 30866...30868 thinking 块 ──
                raw = content
                if "30866" in raw:
                    in_think = True
                if in_think:
                    if "30868" in raw:
                        in_think = False
                        raw = raw.split("30868", 1)[-1]
                    else:
                        raw = ""
                if raw:
                    if ttft_perf is None:
                        ttft_perf = time.perf_counter()
                    chunks.append(raw)
                    end_perf = time.perf_counter()

        # 提取 usage（SGLang 在最后一个 chunk 中返回）
        if chunk.get("usage"):
            u = chunk["usage"]
            usage = {
                "prompt_tokens":     u.get("prompt_tokens"),
                "completion_tokens": u.get("completion_tokens"),
            }

    return chunks, ttft_perf, end_perf, usage


def call_llm(
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 100,
    temperature: float = 0,
    timer: PhasedTimer | None = None,
) -> LlmResult:
    """
    发送 streaming chat completion 请求，返回 LlmResult。
    """
    t = timer or PhasedTimer()
    t0 = time.perf_counter()

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=120,
    )

    chunks, ttft_perf, end_perf, usage = _parse_sse_stream(resp)
    t1 = time.perf_counter()

    text = "".join(chunks).strip()
    ttft_ms = (ttft_perf - t0) * 1000 if ttft_perf else 0
    total_ms = (t1 - t0) * 1000
    decode_ms = total_ms - ttft_ms

    return LlmResult(
        text=text,
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        decode_ms=decode_ms if decode_ms > 0 else 0,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )


def tokenize(base_url: str, model: str, text: str) -> int | None:
    """
    用 SGLang /v1/tokenize 端点精确计 token 数。
    返回 token 数量，失败返回 None。

    用法：获取 system message 和 question 的精确 token 数，
    用 prompt_tokens 减去这两部分得到记忆实际 token 数。
    """
    try:
        resp = requests.post(
            f"{base_url}/v1/tokenize",
            json={"model": model, "prompt": text},
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("count") or len(data.get("tokens", []))
    except Exception:
        pass
    return None
