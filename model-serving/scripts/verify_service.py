#!/usr/bin/env python3
"""Wait for readiness, then save real inference responses for style review."""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8000"
CASES = [
    ("说明概念", "请用三句话解释什么是缓存。"),
    ("改写措辞", "请把这句话改成自然的汉语：我们将以创新为引擎，为团队赋能，打通协作壁垒，打造业务增长的新引擎。"),
    ("避免类比", "请打个比方，向初学者解释什么是机器学习。控制在150字以内。"),
    ("翻译", "翻译成自然的中文：This feature reduces memory usage, but it may slow down text generation."),
]


def main():
    deadline = time.monotonic() + 1800
    while True:
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=10) as r:
                if r.status == 200:
                    break
        except (OSError, urllib.error.URLError):
            pass
        if time.monotonic() > deadline:
            raise SystemExit("服务在30分钟内未就绪；请检查 docker compose logs --tail=100。")
        time.sleep(10)
    results = []
    for name, prompt in CASES:
        payload = {
            "model": "qwen3.8-27b-zh", "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7, "top_p": 0.8, "top_k": 20,
            "presence_penalty": 1.5, "max_tokens": 512, "stream": False,
        }
        start = time.monotonic()
        req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            response = json.load(r)
        choice = response["choices"][0]
        answer = choice["message"]["content"]
        if not answer or not any("\u4e00" <= c <= "\u9fff" for c in answer):
            raise RuntimeError(f"{name} 没有返回中文正文：{response}")
        if choice.get("finish_reason") != "stop":
            raise RuntimeError(f"{name} 未正常结束：{response}")
        results.append({"case": name, "prompt": prompt, "response": response,
                        "seconds": round(time.monotonic() - start, 2)})
        (ROOT / "logs/smoke-test.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"{name}：{answer}", flush=True)
    print("4 次真实推理均成功返回中文正文。文风需结合保存的回答检查。", flush=True)


if __name__ == "__main__":
    main()
