"""Pure retrieval-policy helpers shared by the Task5 runner and tests."""
import re

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "to", "of", "in", "on", "at", "by", "for", "from", "with", "about", "after", "before", "during", "between", "through", "into", "over", "under", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had", "can", "could", "would", "should", "will", "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "many", "much", "please", "tell", "me", "describe", "explain", "list", "name", "identify", "recall", "remember", "according", "mentioned", "say", "said", "happen", "happened", "event", "events", "person", "people", "thing", "things", "fact", "facts", "their", "they", "them", "it", "its", "he", "she", "his", "her", "this", "that", "these", "those", "not", "all", "any", "also", "just", "ever", "ever?"
}


def conditions():
    """Return the fixed, explicit nine-condition retrieval matrix."""
    result = []
    for k in (1, 5, 10, 20):
        result.append({"id": "k%d" % k, "top_k": k, "block_budget": None, "per_record_tokens": None, "query_count": 1})
    result.extend([
        {"id": "b128", "top_k": 20, "block_budget": 128, "per_record_tokens": None, "query_count": 1},
        {"id": "b512", "top_k": 20, "block_budget": 512, "per_record_tokens": None, "query_count": 1},
        {"id": "len24", "top_k": 20, "block_budget": None, "per_record_tokens": 24, "query_count": 1},
        {"id": "k10_b512", "top_k": 10, "block_budget": 512, "per_record_tokens": None, "query_count": 1},
        {"id": "supplement", "top_k": 5, "block_budget": 512, "per_record_tokens": None, "query_count": 2},
    ])
    return result


def supplement_query(question):
    """Create a deterministic keyword query from question text only."""
    tokens = re.findall(r"[\w]+(?:['’-][\w]+)*", question, flags=re.UNICODE)
    keywords = [token for token in tokens if token.casefold() not in _STOP_WORDS]
    if not keywords:
        keywords = tokens
    return " ".join(keywords)


def merge_supplement_results(first, second, limit=10):
    """Keep first-query order, then fill by second-query order, deduping exact text."""
    merged = []
    seen = set()
    for item in list(first) + list(second):
        text = item.get("text", item.get("memory", ""))
        if text in seen:
            continue
        seen.add(text)
        entry = dict(item)
        entry["text"] = text
        entry["rank"] = len(merged) + 1
        merged.append(entry)
        if len(merged) >= limit:
            break
    return merged


def fit_token_budget(text, budget, encode, decode):
    """Return an exact token-prefix decode that re-encodes within the budget."""
    ids = encode(text)
    if budget is None or len(ids) <= budget:
        return text, ids
    cut = budget
    while cut >= 0:
        result = decode(ids[:cut])
        actual = encode(result)
        if len(actual) <= budget:
            return result, actual
        cut -= 1
    raise ValueError("Cannot fit empty text")


def limit_record_text(memories, token_limit, encode, decode):
    """Truncate each retrieved memory separately, before labels are assembled."""
    result = []
    for rank, memory in enumerate(memories, 1):
        entry = dict(memory)
        text, ids = fit_token_budget(entry.get("text", entry.get("memory", "")), token_limit, encode, decode)
        entry["text"] = text
        entry["record_tokens"] = len(ids)
        entry["rank"] = rank
        result.append(entry)
    return result


def assemble_block(memories):
    return "\n".join("[Memory %d] %s" % (i, memory["text"]) for i, memory in enumerate(memories, 1))
