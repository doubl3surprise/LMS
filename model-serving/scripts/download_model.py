#!/usr/bin/env python3
"""Download the official ModelScope snapshot, resume files, and verify SHA-256."""
import concurrent.futures
import hashlib
import json
import os
import resource
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "Qwen3.8-27B"
MANIFEST = ROOT / "downloads" / "modelscope-files.json"


def download(entry):
    name = entry["Path"]
    path = DEST / name
    if path.resolve().parent != DEST.resolve():
        raise ValueError(f"Unexpected nested file: {name}")
    part = path.with_name(path.name + ".part")
    expected = entry["Sha256"]
    if not path.exists():
        url = ("https://modelscope.cn/models/Qwen/Qwen3.8-27B/resolve/"
               + urllib.parse.quote(entry["Revision"], safe="") + "/"
               + urllib.parse.quote(name, safe=""))
        if not part.exists() or part.stat().st_size != entry["Size"]:
            with (ROOT / "logs" / (name + ".download.log")).open("a") as log:
                subprocess.run([
                    "curl", "-fL", "--retry", "5", "--retry-delay", "3",
                    "--connect-timeout", "20", "--speed-limit", "1024",
                    "--speed-time", "120", "-C", "-", "-o", str(part), url,
                ], stderr=log, check=True)
        candidate = part
    else:
        candidate = path
    if candidate.stat().st_size != entry["Size"]:
        raise RuntimeError(f"Size mismatch: {name}")
    with candidate.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch: {name}; remove this file before retrying")
    if candidate == part:
        part.replace(path)
    print(f"Verified {name} ({entry['Size']:,} bytes)", flush=True)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    if not MANIFEST.exists():
        url = "https://modelscope.cn/api/v1/models/Qwen/Qwen3.8-27B/repo/files?Revision=master&Recursive=true"
        with urllib.request.urlopen(url, timeout=60) as response:
            manifest = json.load(response)
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    manifest = json.loads(MANIFEST.read_text())
    entries = [e for e in manifest["Data"]["Files"] if e["Type"] == "blob"
               and e["Path"] not in {"README.md", ".gitattributes", "crc32.txt"}]
    entries.sort(key=lambda e: (e["Size"] >= 20_000_000, e["Path"]))
    total = sum(e["Size"] for e in entries)
    print(f"Downloading {len(entries)} files, {total / 1e9:.2f} GB", flush=True)
    aria = ROOT / "runtime/usr/bin/aria2c"
    if aria.exists():
        hf_manifest_path = ROOT / "downloads/huggingface-files.json"
        hf = json.loads(hf_manifest_path.read_text()) if hf_manifest_path.exists() else {}
        hf_files = {e["rfilename"]: e for e in hf.get("siblings", [])}
        lines = []
        for entry in entries:
            if (DEST / entry["Path"]).exists():
                continue
            url = ("https://modelscope.cn/models/Qwen/Qwen3.8-27B/resolve/"
                   + urllib.parse.quote(entry["Revision"], safe="") + "/"
                   + urllib.parse.quote(entry["Path"], safe=""))
            hf_entry = hf_files.get(entry["Path"], {})
            if hf_entry.get("lfs", {}).get("sha256") == entry["Sha256"]:
                hf_url = ("https://huggingface.co/Qwen/Qwen3.8-27B/resolve/" + hf["sha"]
                          + "/" + urllib.parse.quote(entry["Path"], safe=""))
                url = hf_url + "\t" + url
            lines.extend([url, f"  dir={DEST}", f"  out={entry['Path']}.part",
                          f"  checksum=sha-256={entry['Sha256']}"])
        if lines:
            aria_input = ROOT / "downloads/aria2-input.txt"
            aria_input.write_text("\n".join(lines) + "\n")
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = str(ROOT / "runtime/usr/lib/aarch64-linux-gnu")
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            target = 8192 if hard == resource.RLIM_INFINITY else min(8192, hard)
            if soft < target:
                resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            subprocess.run([str(aria), "--input-file=" + str(aria_input),
                "--continue=true", "--check-integrity=true", "--auto-file-renaming=false",
                "--allow-overwrite=false", "--file-allocation=none",
                "--max-concurrent-downloads=18", "--max-connection-per-server=16", "--split=32",
                "--uri-selector=adaptive", "--min-split-size=1M", "--max-tries=5", "--retry-wait=5", "--connect-timeout=20",
                "--timeout=120", "--summary-interval=30", "--download-result=hide",
                "--show-console-readout=false", "--console-log-level=warn", "--enable-color=false"], env=env, check=True)
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(download, e) for e in entries}
        while futures:
            done, futures = concurrent.futures.wait(futures, timeout=30,
                return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                f.result()
            count = sum(p.stat().st_size for p in DEST.iterdir() if p.is_file())
            print(f"Progress: {count / 1e9:.2f}/{total / 1e9:.2f} GB; "
                  f"elapsed {time.monotonic() - start:.0f}s", flush=True)
    (DEST / "download-verified.json").write_text(json.dumps({
        "source": "https://modelscope.cn/models/Qwen/Qwen3.8-27B",
        "files": [{k: e[k] for k in ("Path", "Size", "Sha256", "Revision")} for e in entries],
    }, ensure_ascii=False, indent=2))
    print("All files verified.", flush=True)


if __name__ == "__main__":
    main()
