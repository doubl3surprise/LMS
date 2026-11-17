#!/usr/bin/env python3
"""Streaming local chat client; Python standard library only."""
import argparse
import http.client
import json
import sys
import urllib.error
import urllib.request


def configure_input():
    """Let GNU readline collect a bracketed paste until the user presses Enter."""
    try:
        import readline
    except ImportError:
        return
    if "GNU readline" in (readline.__doc__ or ""):
        readline.parse_and_bind("set enable-bracketed-paste on")


def read_prompt():
    prompt = input("\n你：").strip()
    if prompt != "/paste":
        return prompt
    print("多行输入：粘贴或输入正文，单独一行 /send 发送，/cancel 取消。")
    lines = []
    while True:
        line = input("… ")
        if line.strip() == "/send":
            return "\n".join(lines).strip()
        if line.strip() == "/cancel":
            print("已取消本次输入。")
            return ""
        lines.append(line)


def error_message(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return f"服务返回 HTTP {exc.code}：{exc.read().decode(errors='replace')}"
    return str(exc)


def chat(args, messages):
    payload = {
        "model": "qwen3.8-27b-zh", "messages": messages, "stream": True,
        "temperature": 1.0 if args.think else 0.7,
        "top_p": 0.95 if args.think else 0.8, "top_k": 20,
        "presence_penalty": 0.0 if args.think else 1.5,
        "max_tokens": args.max_tokens or (8192 if args.think else 2048),
        "chat_template_kwargs": {"enable_thinking": args.think},
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(args.url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    output = []
    finish_reason = None
    saw_reasoning = False
    stream_done = False
    usage = {}
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                stream_done = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                saw_reasoning = saw_reasoning or bool(delta.get("reasoning_content"))
                piece = delta.get("content") or ""
                if piece:
                    print(piece, end="", flush=True)
                    output.append(piece)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    print()
    if not stream_done or finish_reason is None:
        raise RuntimeError("响应流提前中断，未收到完整结束标记；请重试。")
    answer = "".join(output)
    if not answer.strip():
        detail = f"finish_reason={finish_reason}"
        if usage.get("completion_tokens") is not None:
            detail += f"，生成 {usage['completion_tokens']} tokens"
        if finish_reason == "length":
            reason = "思考已耗尽输出额度，尚未生成正文" if saw_reasoning else "已达到输出长度限制，但正文为空"
            hint = "可增大 --max-tokens" + ("，或去掉 --think" if args.think else "") + "。"
        elif saw_reasoning:
            reason = "模型只返回了思考内容，没有生成正文"
            hint = "请重试；若持续出现，请检查服务端思考解析配置。"
        else:
            reason = "模型结束了回复，但正文为空"
            hint = "请提交完整问题；多行内容可用 /paste 输入，多轮对话可先 /clear 再重试。"
        raise RuntimeError(f"{reason}（{detail}）；{hint}")
    if finish_reason == "length":
        print("[已达到输出长度限制，可用 --max-tokens 增大限制。]", file=sys.stderr)
    return answer


def main():
    parser = argparse.ArgumentParser(description="与本地 Qwen3.8-27B 对话（自动应用中文 skill）")
    parser.add_argument("prompt", nargs="*", help="单次提问；省略后进入交互模式")
    parser.add_argument("--think", action="store_true", help="启用模型思考模式")
    parser.add_argument("--max-tokens", type=int, help="最大输出 token 数")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1")
    args = parser.parse_args()
    if args.max_tokens is not None and args.max_tokens < 1:
        parser.error("--max-tokens 必须大于 0")
    if args.prompt or not sys.stdin.isatty():
        prompt = " ".join(args.prompt) if args.prompt else sys.stdin.read().strip()
        if not prompt:
            parser.error("提问内容不能为空")
        chat(args, [{"role": "user", "content": prompt}])
        return
    configure_input()
    print("本地 Qwen3.8-27B。/exit 退出，/clear 清空对话，/paste 输入多行。")
    print("支持粘贴保护的终端可直接粘贴整段文字，再按回车发送。")
    messages = []
    while True:
        try:
            prompt = read_prompt()
        except EOFError:
            break
        if prompt == "/exit":
            break
        if prompt == "/clear":
            messages.clear()
            print("已清空对话。")
            continue
        if not any(char.isprintable() and not char.isspace() for char in prompt):
            continue
        messages.append({"role": "user", "content": prompt})
        print("助手：", end="", flush=True)
        try:
            answer = chat(args, messages)
        except (OSError, http.client.HTTPException, RuntimeError, ValueError) as exc:
            messages.pop()
            print(f"\n请求失败：{error_message(exc)}", file=sys.stderr)
            continue
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    try:
        main()
    except (OSError, http.client.HTTPException, RuntimeError, ValueError) as exc:
        print(f"请求失败：{error_message(exc)}\n请在部署目录运行 ./status.sh 检查服务。", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n已退出。", file=sys.stderr)
        sys.exit(130)
