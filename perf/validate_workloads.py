#!/usr/bin/env python3
"""Controlled metrics-only overhead calibration and real B=1/B=2 retrieval pipelines.

Run inside the benchmark environment. Host resources are intentionally outside the
metrics-only calibration scope; do not run a host sampler during this comparison.
No judge or reference answer enters the timed work.
"""
import argparse
import hashlib
import json
import math
import random
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Lock


def percentile(values, p):
    """Linear interpolation at (n-1)*p/100; empty measurements stay missing."""
    if not 0 <= p <= 100:
        raise ValueError('percentile must be between 0 and 100')
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return None
    index = (len(values) - 1) * p / 100
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def delta(baseline, observed):
    absolute = observed - baseline if baseline is not None and observed is not None else None
    return {'absolute': absolute, 'percent': absolute / baseline * 100 if absolute is not None and baseline else None}


def compare_samples(samples, fields):
    """Descriptive paired comparisons only; never infer significance."""
    indexed = {}
    for sample in samples:
        key = (sample['repetition'], sample['condition'])
        if key in indexed:
            raise ValueError('duplicate repetition/condition')
        indexed[key] = sample
    output = {}
    for field in fields:
        summaries = {}
        for condition in ('baseline', 'metrics_only'):
            values = [s.get(field) for s in samples if s['condition'] == condition]
            values = [v for v in values if v is not None]
            summaries[condition] = {'n': len(values), 'mean': statistics.mean(values) if values else None,
                                    'p50': percentile(values, 50), 'p95': percentile(values, 95)}
        pairs = []
        for repetition in sorted({s['repetition'] for s in samples}):
            a = indexed.get((repetition, 'baseline'), {}).get(field)
            b = indexed.get((repetition, 'metrics_only'), {}).get(field)
            pairs.append({'repetition': repetition, 'baseline': a, 'metrics_only': b, 'delta': delta(a, b)})
        output[field] = {**summaries, 'mean_delta': delta(summaries['baseline']['mean'], summaries['metrics_only']['mean']), 'pairs': pairs}
    return output


def overlap_summary(events):
    """Sweep active phases by distinct request; waiting/envelopes are excluded."""
    points = {}
    for event in events:
        start, end = event['start'], event['end']
        if end < start:
            raise ValueError('invalid event interval')
        if event['phase'] in ('request', 'lock_wait') or start == end:
            continue
        points.setdefault(start, []).append((event['request_id'], 1))
        points.setdefault(end, []).append((event['request_id'], -1))
    active, previous, overlap, maximum = {}, None, 0.0, 0
    for stamp in sorted(points):
        if previous is not None and len(active) > 1:
            overlap += stamp - previous
        for rid, change in points[stamp]:
            active[rid] = active.get(rid, 0) + change
            if active[rid] == 0:
                del active[rid]
        maximum = max(maximum, len(active))
        previous = stamp
    return {'overlap_ms': overlap * 1000, 'max_active_requests': maximum, 'cross_request_overlap': overlap > 0}


def run_calibration(backend, out, max_tokens=16, seed=42):
    """Five paired repetitions; same IDs, max tokens and warmed cache throughout."""
    from trace_core_v4 import native_timing
    cfg = backend.cfg
    messages = [{'role': 'user', 'content': 'List twenty interesting scientific facts, numbered 1 through 20. Explain each fact in a complete sentence.'}]
    ids = backend.tokenize(cfg.sglang_url, cfg.model, messages=messages)
    digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    rng = random.Random(seed)
    first = ['baseline', 'metrics_only']
    rng.shuffle(first)
    orders = [first[:] if i % 2 == 0 else first[::-1] for i in range(5)]
    run_id = uuid.uuid4().hex
    samples, pairs, warmups = [], [], []
    engine_before = backend.snapshot()
    for repetition, order in enumerate(orders):
        pairs.append({'repetition': repetition, 'order': order})
        for condition in order:
            sampler = backend.sampler(Path(out) / f'calibration_metrics_{repetition}.jsonl') if condition == 'metrics_only' else None
            try:
                if sampler:
                    sampler.start()
                # Same one-request warmup for each arm, excluded from samples.
                warm = backend.generate(cfg.sglang_url, ids, f'{run_id}:warm:{repetition}:{condition}', max_tokens)
                warmups.append({'repetition': repetition, 'condition': condition, 'completion_tokens': warm['meta'].get('completion_tokens'), 'native_meta': warm['meta']})
                start = time.perf_counter()
                result = backend.generate(cfg.sglang_url, ids, f'{run_id}:measure:{repetition}:{condition}', max_tokens)
                elapsed = (time.perf_counter() - start) * 1000
            finally:
                if sampler:
                    sampler.stop()
            samples.append({'repetition': repetition, 'condition': condition, 'response_ms': elapsed,
                            **native_timing(result['meta'], result['ttft_ms']),
                            'decode_ms': result['decode_ms'], 'generate_total_ms': result['total_ms'],
                            'completion_tokens': result['meta'].get('completion_tokens'),
                            'prompt_tokens': result['meta'].get('prompt_tokens'),
                            'cached_tokens': result['meta'].get('cached_tokens'),
                            'input_ids_sha256': digest, 'answer': result['text'], 'native_meta': result['meta']})
    fields = ['response_ms', 'server_prefill_elapsed_ms', 'server_queue_ms', 'decode_ms', 'completion_tokens']
    return {'scope': 'MetricsSampler interval=0.1 seconds only; no host resource sampler',
            'repetitions_per_condition': 5, 'max_tokens': max_tokens, 'seed': seed,
            'messages': messages, 'input_ids': ids, 'input_ids_sha256': digest,
            'pairs': pairs, 'warmups': warmups, 'samples': samples,
            'comparison': compare_samples(samples, fields),
            'all_outputs_reached_max_tokens': all(s['completion_tokens'] == max_tokens for s in samples),
            'engine_before': engine_before, 'engine_after': backend.snapshot(),
            'limitations': ['Descriptive differences only; five pairs do not establish statistical significance.',
                            'EOS is honored; compare actual completion tokens and the output-length check before interpreting latency.',
                            'Identical IDs intentionally warm prefix caches; cached_tokens are retained.',
                            'Sampler startup and teardown are excluded; sampler runs during each warmup and timed request.',
                            'No host sampler is launched or controlled; external monitoring must be disabled for a clean metrics-only contrast.']}


def run_pipeline_condition(backend, questions, batch_size, repetition, max_tokens):
    """At most B in-flight requests; each performs live retrieval then native generation."""
    from trace_core_v4 import build_prompt, native_timing
    if batch_size not in (1, 2) or len(questions) != 4:
        raise ValueError('pipeline requires B=1 or B=2 and exactly four questions')
    cfg = backend.cfg
    retrieval_lock, event_lock = Lock(), Lock()
    events = []
    condition_id = f'{uuid.uuid4().hex}:r{repetition}:B{batch_size}'
    engine_before = backend.snapshot()

    @contextmanager
    def phase(rid, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            event = {'condition_id': condition_id, 'repetition': repetition, 'batch_size': batch_size,
                     'request_id': rid, 'phase': name, 'start': start, 'end': time.perf_counter()}
            with event_lock:
                events.append(event)

    class EventTimer:
        # Keep the installed PhasedTimer as the source of phase milliseconds.
        def __init__(self, rid):
            self.rid, self.timer = rid, backend.Timer()

        @contextmanager
        def phase(self, name):
            with phase(self.rid, name), self.timer.phase(name):
                yield

        def ms(self, name):
            return self.timer.ms(name)

    def worker(item):
        source_idx, question = item
        rid = f'{condition_id}:q{source_idx}'
        timer = EventTimer(rid)
        wall_start, start = time.time(), time.perf_counter()
        cpu_start = time.thread_time()
        with phase(rid, 'request'):
            with phase(rid, 'lock_wait'):
                retrieval_lock.acquire()
            try:
                memories, vector = backend.retrieve(question['question'], timer)
            finally:
                retrieval_lock.release()
            with timer.phase('assembly'):
                prompt = build_prompt(question['question'], [vars(m).copy() for m in memories])
            with timer.phase('tokenize'):
                ids = backend.tokenize(cfg.sglang_url, cfg.model, messages=prompt['messages'])
            with phase(rid, 'generate'):
                result = backend.generate(cfg.sglang_url, ids, rid, max_tokens)
            response_ms = (time.perf_counter() - start) * 1000
            thread_cpu_ms = (time.thread_time() - cpu_start) * 1000
        timing = {name + '_ms': timer.ms(name) for name in ('embed', 'search', 'mmr', 'assembly', 'tokenize')}
        timing.update(native_timing(result['meta'], result['ttft_ms']))
        timing.update(response_ms=response_ms, ttft_ms=result['ttft_ms'], decode_ms=result['decode_ms'], generate_total_ms=result['total_ms'])
        return {'request_id': rid, 'condition_id': condition_id, 'source_q_idx': source_idx,
                'repetition': repetition, 'batch_size': batch_size, 'question': question['question'],
                'category': question.get('category'), 'wall_start': wall_start, 'timing': timing,
                'memories': prompt['memories'], 'messages': prompt['messages'],
                'memory_block': prompt['memory_block'], 'query_vector': vector.tolist() if hasattr(vector, 'tolist') else vector,
                'input_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                'input_ids': ids, 'answer': result['text'], 'native_meta': result['meta'],
                'tokens': {key: result['meta'].get(key) for key in ('prompt_tokens', 'completion_tokens', 'cached_tokens')},
                'client_token_arrivals': result.get('arrivals', []),
                'local_cpu': {'thread_cpu_ms': thread_cpu_ms, 'scope': 'current worker thread; excludes engine and other threads'}}

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        results = list(pool.map(worker, questions))
    elapsed = time.perf_counter() - start
    events.sort(key=lambda e: (e['start'], e['request_id'], e['phase']))
    return {'condition_id': condition_id, 'batch_size': batch_size, 'repetition': repetition,
            'elapsed_ms': elapsed * 1000, 'request_count': len(results),
            'throughput_requests_per_second': len(results) / elapsed,
            'response_p50_ms': percentile([r['timing']['response_ms'] for r in results], 50),
            'response_p95_ms': percentile([r['timing']['response_ms'] for r in results], 95),
            'engine_before': engine_before, 'engine_after': backend.snapshot(),
            'overlap': overlap_summary(events), 'events': events, 'results': results}


class LiveBackend:
    """Adapter to the installed v4 benchmark stack; heavy imports are CLI-only."""
    def __init__(self, platform):
        import run_trace_v4 as native
        from tracer.config import get_config
        from tracer.retriever import retrieve, get_embedder
        from tracer.timer import PhasedTimer
        from tracer.observability_v4 import MetricsSampler
        from qdrant_client import QdrantClient
        self.cfg = get_config(platform)
        self.generate, self.tokenize = native.generate, native.tokenize
        self.Timer, self._retrieve = PhasedTimer, retrieve
        self._get_embedder, self._Sampler = get_embedder, MetricsSampler
        self.qc = QdrantClient(url=self.cfg.qdrant_url) if self.cfg.qdrant_mode == 'server' else QdrantClient(path=self.cfg.qdrant_path)

    def snapshot(self):
        import requests
        response = requests.get(self.cfg.sglang_url + '/get_server_info', timeout=15)
        response.raise_for_status()
        return {'wall_time': time.time(), 'server_info': response.json()}

    def retrieve(self, question, timer):
        cfg = self.cfg
        return self._retrieve(question, self.qc, cfg.collection, cfg.user_id,
                              cfg.cand_k, cfg.final_k, cfg.mmr_lambda, timer)

    def warmup(self, question):
        self._get_embedder()
        self.retrieve(question, self.Timer())

    def sampler(self, path):
        return self._Sampler(self.cfg.sglang_url, path, interval=.1)

    def close(self):
        self.qc.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', choices=['dgx', '5880ada'], required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--repetitions', type=int, default=3, help='pipeline repetitions; calibration always uses five pairs')
    parser.add_argument('--max-tokens', type=int, default=16)
    parser.add_argument('--conv', type=int, default=0)
    parser.add_argument('--question-indices', nargs=4, type=int, default=[0, 1, 2, 3], metavar='IDX', help='four exact qa indices in the selected dataset conversation')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if args.repetitions < 1 or args.max_tokens < 1:
        parser.error('repetitions and max-tokens must be positive')
    if args.conv < 0 or min(args.question_indices) < 0 or len(set(args.question_indices)) != 4:
        parser.error('conv and question-indices must be nonnegative; question-indices must be distinct')
    return args


def run(args, backend=None):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation prevents accidental replacement of prior evidence.
    target = out / 'experimental.json'
    with target.open('x') as handle:
        json.dump({'status': 'running'}, handle)
    document = {'schema_version': 'workload-validation-1.0', 'status': 'running',
                'metadata': {'arguments': vars(args), 'clock': {'wall_time': time.time(), 'perf_counter': time.perf_counter()},
                             'event_clock': 'time.perf_counter seconds in the same client process'},
                'calibration': None, 'pipeline': {'conditions': [], 'events': [], 'results': []},
                'limitations': ['Metrics-only calibration excludes host resource collection overhead; host samplers must be disabled externally.',
                                'Retrieval is serialized with one lock for embedded Qdrant and shared embedder safety on both platforms.',
                                'B is the client in-flight bound, not a guarantee of engine batching or GPU kernel concurrency.',
                                'Events measure client elapsed phases; server prefill elapsed is scheduler wall time, not summed GPU kernel time.',
                                'Thread CPU is worker-local; process-wide CPU deltas are not attributed to concurrent requests.',
                                'No judge is run; short outputs may stop on EOS before max_tokens.',
                                'Warm prefix caches and fixed question order affect results; all native cache counters are retained.']}

    def persist():
        temporary = out / 'experimental.json.partial'
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2))
        temporary.replace(target)

    try:
        backend = backend if backend is not None else LiveBackend(args.platform)
        cfg = backend.cfg
        document['metadata']['config'] = vars(cfg)
        dataset = Path(cfg.dataset_path).read_bytes()
        document['metadata']['dataset_sha256'] = hashlib.sha256(dataset).hexdigest()
        qa = json.loads(dataset)[args.conv]['qa']
        questions = [(i, qa[i]) for i in args.question_indices]
        if any(not isinstance(q.get('question'), str) or not q['question'].strip() for _, q in questions):
            raise ValueError('each selected qa item must have a nonempty question')
        document['metadata']['questions'] = [{'source_q_idx': i, 'question': q['question'], 'category': q.get('category')} for i, q in questions]
        document['metadata']['engine_before'] = backend.snapshot()
        document['calibration'] = run_calibration(backend, out, args.max_tokens, args.seed)
        persist()
        # Load the singleton embedder and warm all selected retrievals before timing.
        for _, question in questions:
            backend.warmup(question['question'])
        rng = random.Random(args.seed)
        first = [1, 2]
        rng.shuffle(first)
        for repetition in range(args.repetitions):
            order = first if repetition % 2 == 0 else first[::-1]
            for batch_size in order:
                condition = run_pipeline_condition(backend, questions, batch_size, repetition, args.max_tokens)
                document['pipeline']['events'].extend(condition.pop('events'))
                document['pipeline']['results'].extend(condition.pop('results'))
                document['pipeline']['conditions'].append(condition)
                persist()
        pipeline = document['pipeline']
        pipeline['paired_deltas_B2_minus_B1'] = []
        fields = ['elapsed_ms', 'throughput_requests_per_second', 'response_p50_ms', 'response_p95_ms']
        for repetition in range(args.repetitions):
            arms = {c['batch_size']: c for c in pipeline['conditions'] if c['repetition'] == repetition}
            pipeline['paired_deltas_B2_minus_B1'].append({'repetition': repetition, **{f: delta(arms[1][f], arms[2][f]) for f in fields}})
        pipeline['summary_by_batch_size'] = {str(b): {f: statistics.mean(c[f] for c in pipeline['conditions'] if c['batch_size'] == b) for f in fields} for b in (1, 2)}
        signatures = {i: {r['input_ids_sha256'] for r in pipeline['results'] if r['source_q_idx'] == i} for i, _ in questions}
        pipeline['checks'] = {'request_count_matches': len(pipeline['results']) == args.repetitions * 2 * 4,
                              'same_input_ids_across_conditions': all(len(s) == 1 for s in signatures.values()),
                              'B1_no_cross_request_overlap': all(not c['overlap']['cross_request_overlap'] for c in pipeline['conditions'] if c['batch_size'] == 1),
                              'B2_cross_request_overlap_each_repetition': all(c['overlap']['cross_request_overlap'] for c in pipeline['conditions'] if c['batch_size'] == 2)}
        acceptance_checks = {**pipeline['checks'],
                             'calibration_outputs_reached_max_tokens': document['calibration']['all_outputs_reached_max_tokens']}
        document['acceptance'] = {'checks': acceptance_checks, 'passed': all(acceptance_checks.values()),
                                  'scope': 'input comparability, output length, counts and observed client overlap; no speedup or significance threshold'}
        document['metadata']['engine_after'] = backend.snapshot()
        document['status'] = 'complete'
    except BaseException as error:
        document['status'] = 'failed'
        document['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        try:
            if backend is not None:
                backend.close()
        except BaseException as error:
            document['status'] = 'failed'
            document['close_error'] = {'type': type(error).__name__, 'message': str(error)}
            raise
        finally:
            persist()
    print('EXPERIMENTAL_JSON=' + str(target.resolve()), flush=True)
    return document


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
