# Task 4 memory span audit

Span ranges are inclusive-start/exclusive-end token indexes into the exact prompt IDs. A request is verified only when decoding those IDs with the DGX tokenizer, re-encoding the rendered text without added special tokens yields identical IDs, and each memory's raw text is uniquely located in the final user message, or disambiguated by its unique recorded `[Memory rank] ` label at a line boundary. Token offsets overlapping memory characters define the range; no inferred template offsets are used.

| Host input | Cohort | Requests | Verified | Unverified |
|---|---:|---:|---:|---:|
| dgx | native32 | 96 | 96 | 0 |
| dgx | mmr152 | 152 | 152 | 0 |
| dgx | sessions | 9 | 9 | 0 |
| 5880ada | native32 | 96 | 96 | 0 |
| 5880ada | mmr152 | 152 | 152 | 0 |
| 5880ada | sessions | 9 | 9 | 0 |

All 28 initially ambiguous rows (14 per host) are now verified using unique recorded `[Memory rank] ` labels at line boundaries. No raw-text guessing is used. The current collector verifies all 257 requests per host; the native32 analysis selects only the 32 native retrieval requests, all verified. All final memory ID/hash/token bounds are asserted by analyze.py.

Artifacts: `collect_memory_spans.py`, `test_memory_spans.py`, `inputs/dgx/memory_spans.json`, and `inputs/5880ada/memory_spans.json`. Tokenizer used: `/home/jiangchuan/mem0-reproduction/model-serving/experiment/tokenizer/tokenizer.json` with the DGX reproduction `.venv` (`tokenizers 0.23.3`).

Verification: `python3 -m unittest discover -v` passed all 25 task tests. Both output files contain all 257 requested entries per host. No source traces, prompt-ID manifests, tokenizer, or model data were modified.
