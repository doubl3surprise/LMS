#!/usr/bin/env python3
"""Summarize resumable downloads without counting sparse holes as data."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
model = ROOT / "models/Qwen3.8-27B"
manifest = ROOT / "downloads/modelscope-files.json"
total = sum(e["Size"] for e in json.loads(manifest.read_text())["Data"]["Files"]
            if e["Type"] == "blob" and e["Path"] not in {"README.md", ".gitattributes", "crc32.txt"})
if (model / "download-verified.json").exists():
    print(f"权重已下载并校验：{total / 1e9:.2f} GB")
else:
    allocated = sum(p.stat().st_blocks * 512 for p in model.iterdir() if p.is_file())
    log = (ROOT / "logs/model-download.log").read_text(errors="replace")
    summary = log.split("*** Download Progress Summary")[-1]
    speeds = re.findall(r"DL:([\d.]+)([KMG]iB)", summary)
    rate = sum(float(n) * {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024}[u] for n, u in speeds)
    print(f"已写入约 {allocated / 1e9:.2f}/{total / 1e9:.2f} GB（{allocated / total:.1%}），最近汇总速度 {rate:.2f} MiB/s")
    if "ERROR" in log:
        print(f"下载日志累计 {log.count('[ERROR]')} 条错误记录（连接重试也会记录），详见 logs/model-download.log。")
print("推理验证结果：" + ("已生成 logs/smoke-test.json" if (ROOT / "logs/smoke-test.json").exists() else "尚未生成"))
