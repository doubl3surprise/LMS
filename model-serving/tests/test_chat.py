"""Run with python3 -m unittest discover -s tests -v (no model required)."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import queue
import select
import subprocess
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import chat


def stream(*events, done=True):
    data = b"".join(
        ("data: " + json.dumps(event) + "\n\n").encode() for event in events
    )
    return data + (b"data: [DONE]\n\n" if done else b"")


def chunk(content=None, reasoning=None, finish=None):
    return {"choices": [{"delta": {"content": content, "reasoning_content": reasoning},
                         "finish_reason": finish}]}


class StreamTests(unittest.TestCase):
    def run_chat(self, response, think=False):
        args = argparse.Namespace(think=think, max_tokens=32, url="http://local/v1")
        with patch("chat.urllib.request.urlopen", return_value=io.BytesIO(response)):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return chat.chat(args, [{"role": "user", "content": "问题"}])

    def test_content_and_usage_after_reasoning(self):
        result = self.run_chat(stream(
            chunk(reasoning="思考"), chunk("正文"), chunk("。"), chunk(finish="stop"),
            {"choices": [], "usage": {"completion_tokens": 8}},
        ), think=True)
        self.assertEqual(result, "正文。")

    def test_empty_stop_reports_cause_without_token_advice(self):
        with self.assertRaises(RuntimeError) as error:
            self.run_chat(stream(chunk(finish="stop"),
                                 {"usage": {"completion_tokens": 1}}))
        self.assertIn("finish_reason=stop", str(error.exception))
        self.assertIn("生成 1 tokens", str(error.exception))
        self.assertNotIn("--max-tokens", str(error.exception))

    def test_reasoning_exhausts_budget(self):
        with self.assertRaisesRegex(RuntimeError, "思考已耗尽.*--max-tokens.*--think"):
            self.run_chat(stream(chunk(reasoning="思考"), chunk(finish="length")), think=True)

    def test_empty_length_without_reasoning(self):
        with self.assertRaisesRegex(RuntimeError, "已达到输出长度限制.*--max-tokens"):
            self.run_chat(stream(chunk(finish="length")))

    def test_whitespace_is_not_an_answer(self):
        with self.assertRaisesRegex(RuntimeError, "正文为空"):
            self.run_chat(stream(chunk(" \n"), chunk(finish="stop")))

    def test_truncated_content_is_preserved(self):
        self.assertEqual(self.run_chat(stream(chunk("部分正文"), chunk(finish="length"))), "部分正文")

    def test_incomplete_stream_is_rejected(self):
        for response in (stream(chunk("部分正文"), done=False),
                         stream(chunk("正文"), chunk(finish="stop"), done=False),
                         stream(chunk("正文"))):
            with self.subTest(response=response), self.assertRaisesRegex(RuntimeError, "响应流提前中断"):
                self.run_chat(response)

    def test_server_error_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, "service unavailable"):
            self.run_chat(stream({"error": {"message": "service unavailable"}}))


class InputTests(unittest.TestCase):
    def test_multiline_keeps_internal_whitespace_and_commands(self):
        with patch("builtins.input", side_effect=["/paste", "第一行", "", "  缩进", "/exit", "/send"]):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(chat.read_prompt(), "第一行\n\n  缩进\n/exit")

    def test_cancel_discards_draft(self):
        with patch("builtins.input", side_effect=["/paste", "草稿", "/cancel"]):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(chat.read_prompt(), "")

    def test_failed_turn_is_removed_from_history(self):
        requests = []

        def respond(args, messages):
            requests.append(list(messages))
            if len(requests) == 2:
                raise RuntimeError("空回复")
            return "回答"

        with patch.object(sys, "argv", ["chat.py"]), patch("sys.stdin.isatty", return_value=True):
            with patch("chat.configure_input"), patch("builtins.input", side_effect=["第一问", "失败的问题", "第三问", "/exit"]):
                with patch("chat.chat", side_effect=respond), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    chat.main()
        self.assertEqual(requests[2], [
            {"role": "user", "content": "第一问"},
            {"role": "assistant", "content": "回答"},
            {"role": "user", "content": "第三问"},
        ])


@unittest.skipUnless(sys.platform.startswith("linux"), "GNU readline terminal regression")
class TerminalTests(unittest.TestCase):
    def test_bracketed_paste_sends_one_request_only_after_enter(self):
        import pty
        import readline
        if "GNU readline" not in (readline.__doc__ or ""):
            self.skipTest("GNU readline unavailable")
        requests = queue.Queue()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.put(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                data = stream(chunk("已收到。"), chunk(finish="stop"))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        master, slave = pty.openpty()
        proc = subprocess.Popen(
            [sys.executable, str(Path(chat.__file__).resolve()), "--url",
             f"http://127.0.0.1:{server.server_port}/v1"],
            stdin=slave, stdout=slave, stderr=slave,
            env={**os.environ, "TERM": "xterm-256color", "INPUTRC": "/dev/null"},
        )
        os.close(slave)

        def wait_for(text):
            received = b""
            deadline = time.monotonic() + 5
            while text.encode() not in received:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([master], [], [], remaining)[0]:
                    self.fail(f"Terminal did not print {text!r}: {received!r}")
                received += os.read(master, 65536)

        try:
            wait_for("你：")
            prompt = "整理下面的文字：\n第一段\n\n第二段"
            os.write(master, ("\x1b[200~" + prompt + "\x1b[201~").encode())
            with self.assertRaises(queue.Empty):
                requests.get(timeout=0.2)
            os.write(master, b"\r")
            payload = requests.get(timeout=5)
            self.assertEqual(payload["messages"], [{"role": "user", "content": prompt}])
            wait_for("已收到。")
            os.write(master, b"/exit\r")
            proc.wait(timeout=5)
            self.assertEqual(proc.returncode, 0)
            self.assertTrue(requests.empty())
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            os.close(master)
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
