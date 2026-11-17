"""Map exact memory text in the final user message to frozen prompt token IDs."""
import argparse
import hashlib
import json
from pathlib import Path


def _failure(reason):
    return {"verified": False, "failure": reason, "spans": []}


def _find_unique(haystack, needle):
    if not needle:
        return None
    first = haystack.find(needle)
    if first < 0 or haystack.find(needle, first + 1) >= 0:
        return None
    return first, first + len(needle)


def collect_request(row, prompt_ids, tokenizer):
    """Return token spans only after exact prompt decode/encode round-trip."""
    try:
        rendered = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        encoding = tokenizer.encode(rendered, add_special_tokens=False)
    except Exception as exc:
        return _failure("tokenizer_error: " + type(exc).__name__)
    if list(encoding.ids) != list(prompt_ids):
        return _failure("roundtrip_token_ids_mismatch")
    if len(encoding.offsets) != len(prompt_ids):
        return _failure("offset_count_mismatch")

    messages = row.get("messages") or []
    if not messages or messages[-1].get("role") != "user":
        return _failure("missing_final_user_message")
    current = messages[-1].get("content")
    if not isinstance(current, str):
        return _failure("final_user_content_not_text")
    user_match = _find_unique(rendered, current)
    if user_match is None:
        return _failure("final_user_message_not_unique_in_rendered_prompt")
    user_start, user_end = user_match

    spans = []
    for memory in row.get("memories", []):
        memory_id = memory.get("memory_id", memory.get("id"))
        text = memory.get("text")
        if memory_id is None or not isinstance(text, str):
            return _failure("memory_missing_id_or_text")
        occurrences = []
        start = 0
        while True:
            pos = current.find(text, start)
            if pos < 0:
                break
            occurrences.append(pos)
            start = pos + 1
        if len(occurrences) != 1:
            rank = memory.get('rank')
            labelled = '[Memory '+str(rank)+'] '+text
            match = _find_unique(current,labelled) if rank is not None else None
            if match is None or (match[0] > 0 and current[match[0]-1] != '\n'):
                return _failure("memory_text_not_unique_in_final_user_message:" + str(memory_id))
            occurrences = [match[0] + len(labelled) - len(text)]
        char_start = user_start + occurrences[0]
        char_end = char_start + len(text)
        if char_start < user_start or char_end > user_end:
            return _failure("memory_outside_final_user_message:" + str(memory_id))
        touched = [i for i, (a, b) in enumerate(encoding.offsets)
                   if b > char_start and a < char_end]
        if not touched:
            return _failure("memory_has_no_token_offsets:" + str(memory_id))
        spans.append({
            "memory_id": str(memory_id),
            "start": touched[0],
            "end": touched[-1] + 1,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
    return {"verified": True, "failure": None, "spans": spans}


def load_tokenizer(path):
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("install tokenizers in the selected Python environment") from exc
    return Tokenizer.from_file(str(path))


def collect(root, tokenizer_path, cohorts=None):
    root = Path(root)
    tokenizer = load_tokenizer(tokenizer_path)
    selected = cohorts or {"native32": "native32_trace.jsonl", "mmr152": "mmr152_trace.jsonl", "sessions": "sessions_trace.jsonl"}
    output = {}
    for cohort, trace_name in selected.items():
        ids = json.loads((root / "prompt_ids.json").read_text())[cohort]
        rows = {}
        with (root / trace_name).open() as stream:
            traces = {row["request_id"]: row for row in map(json.loads, stream)}
        for request_id, prompt in ids.items():
            row = traces.get(request_id)
            if row is None:
                rows[request_id] = _failure("request_id_missing_from_trace")
                continue
            actual_ids = prompt.get("token_ids")
            if not isinstance(actual_ids, list):
                rows[request_id] = _failure("prompt_token_ids_missing")
                continue
            rows[request_id] = collect_request(row, actual_ids, tokenizer)
        output[cohort] = rows
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="cohort input directory")
    parser.add_argument("--tokenizer", required=True, help="exact tokenizer.json used for prompts")
    parser.add_argument("--output", required=True, help="memory_spans.json output path")
    args = parser.parse_args()
    data = collect(args.root, args.tokenizer)
    Path(args.output).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    counts = {cohort: {"requests": len(rows), "verified": sum(r["verified"] for r in rows.values())}
              for cohort, rows in data.items()}
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
