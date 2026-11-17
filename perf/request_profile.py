#!/usr/bin/env python3
"""Replay trace messages in separate, manually stopped SGLang profiling windows.

Attribution is to an otherwise idle profiling window, not a CUDA correlation-ID
proof of request ownership. Requires external exclusion of other client traffic.
"""
import argparse
import json
import subprocess
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

from profile_trace_v4 import check_idle_metrics, parse_traces

LIMITATIONS = [
    'Replay of saved messages, not instrumentation of the original request; cache state and output length may differ.',
    'exclusive_profile_window requires external exclusion of other inference clients; idle gauges are a preflight check, not a server lock.',
    'GPU kernels belong to the profile window; no per-kernel request correlation is proved. Background health/heartbeat work may be included.',
    'No metrics, health or server-info polling is issued by this script inside the window; independent monitors may still contaminate it.',
    'CPU deltas cover the SGLang process tree from before start_profile through after stop_profile, including profiler overhead, idle/background work and export.',
    'CPU snapshots omit processes exiting between snapshots and do not give per-rank CPU ownership or Python operator self-time.',
    'GPU kernel union is per device/rank/process; do not sum unions across ranks as request wall latency. DRAM bandwidth is not measured.',
    'No cache flush or server restart is performed; this is a bounded replay, with max_new_tokens defaulting to 16.',
]

CPU_PROBE = r'''import json, os, time
from pathlib import Path
rows = {}
self_pid = os.getpid()
hz = os.sysconf('SC_CLK_TCK')
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name) == self_pid:
        continue
    try:
        raw = (p / 'stat').read_text()
        fields = raw[raw.rindex(')') + 2:].split()
        cmd = (p / 'cmdline').read_bytes().replace(b'\x00', b' ').decode(errors='replace')
        rows[int(p.name)] = {'pid': int(p.name), 'ppid': int(fields[1]),
            'start_ticks': int(fields[19]), 'user_s': int(fields[11]) / hz,
            'system_s': int(fields[12]) / hz, 'cmd': cmd}
    except (OSError, ValueError, IndexError):
        pass
roots = {pid for pid, r in rows.items() if 'sglang' in r['cmd'].lower()}
selected = set(roots)
while True:
    expanded = selected | {pid for pid, r in rows.items() if r['ppid'] in selected}
    if expanded == selected:
        break
    selected = expanded
print(json.dumps({'processes': [{k: v for k, v in rows[pid].items() if k != 'cmd'}
    for pid in sorted(selected)], 'roots': sorted(roots), 'clock_ticks_per_second': hz,
    'monotonic_ns': time.monotonic_ns()}))
'''


def http_json(url, payload=None, timeout=15):
    body = None if payload is None else json.dumps(payload).encode('utf-8')
    request = Request(url, data=body, headers={'Content-Type': 'application/json'} if body is not None else {})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(8_000_001)
    if len(raw) > 8_000_000:
        raise RuntimeError('HTTP response exceeded 8 MB')
    text = raw.decode('utf-8')
    if payload is None:
        return text
    try:
        value = json.loads(text) if text else {}
    except ValueError:
        value = text
    if isinstance(value, dict) and (value.get('success') is False or value.get('error')):
        raise RuntimeError('server rejected request: ' + str(value))
    return value


def run_command(argv, timeout=30):
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    if completed.returncode:
        raise RuntimeError('command failed (%s): %s' % (completed.returncode, completed.stderr[-4000:]))
    if len(completed.stdout) > 4_000_000:
        raise RuntimeError('command output exceeded 4 MB')
    return completed.stdout


def cpu_delta(before, after):
    result = {'status': 'unavailable', 'user_ms': None, 'system_ms': None,
              'scope': 'SGLang process tree; before start_profile through after stop_profile',
              'profiler_overhead_included': True, 'before': before, 'after': after}
    if before.get('error') or after.get('error'):
        result['error'] = 'CPU snapshot failed'
        return result
    left = {(p['pid'], p['start_ticks']): p for p in before.get('processes', [])}
    right = {(p['pid'], p['start_ticks']): p for p in after.get('processes', [])}
    common = left.keys() & right.keys()
    result.update({'matched_process_count': len(common),
                   'lost_processes': [list(k) for k in sorted(left.keys() - right.keys())],
                   'new_processes': [list(k) for k in sorted(right.keys() - left.keys())]})
    if not common:
        result['error'] = 'No stable SGLang processes across snapshots'
        return result
    deltas = [(right[k]['user_s'] - left[k]['user_s'], right[k]['system_s'] - left[k]['system_s']) for k in common]
    if any(u < 0 or s < 0 for u, s in deltas):
        result['error'] = 'CPU counters decreased'
        return result
    result.update({'status': 'partial' if left.keys() != right.keys() else 'success',
                   'user_ms': sum(u for u, s in deltas) * 1000,
                   'system_ms': sum(s for u, s in deltas) * 1000})
    return result


def snapshot(container, command):
    try:
        return json.loads(command(['docker', 'exec', '--user', '0', container, 'python3', '-c', CPU_PROBE], timeout=15))
    except Exception as exc:
        return {'error': str(exc), 'processes': []}


def collect(args, http=http_json, command=run_command):
    selected = load_requests(args.requests, args.limit)
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = out / 'request_profiles.json'
    if summary.exists():
        raise FileExistsError('refusing to overwrite ' + str(summary))
    run_id = 'request-profiles-' + uuid.uuid4().hex
    artifacts = out / run_id
    artifacts.mkdir(exist_ok=False)
    base = args.base_url.rstrip('/')
    result = {'status': 'error', 'platform': args.platform, 'run_id': run_id,
              'requests_path': str(Path(args.requests).resolve()), 'container': args.container,
              'base_url': base, 'selected_count': len(selected), 'requests': [],
              'limitations': LIMITATIONS[:], 'errors': []}
    for index, source in enumerate(selected):
        profile_id = str(uuid.uuid4())
        remote = '/root/.cache/task2_profiles/' + run_id + '/' + profile_id
        local = artifacts / profile_id
        rid = 'profile-replay-' + profile_id
        record = {'request_id': source['request_id'], 'source_line': source['source_line'],
                  'replay_request_id': rid, 'profile_id': profile_id, 'remote_output_dir': remote,
                  'source': 'exclusive_profile_window', 'profile_status': 'error',
                  'artifact_paths': [], 'artifact_map': [], 'errors': [], 'native_meta': None,
                  'generation_status': 'not_started', 'profile_stopped': False}
        result['requests'].append(record)
        attempted = False
        before, after = {'error': 'window not started'}, {'error': 'window not stopped'}
        try:
            idle = check_idle_metrics(http(base + '/metrics', timeout=15))
            record['queue_check'] = idle
            if not idle['idle']:
                raise RuntimeError('refusing busy or unverifiable server: ' + str(idle['reason']))
            tokens = http(base + '/v1/tokenize', {'model': args.model,
                'messages': source['messages'], 'add_special_tokens': False,
                'chat_template_kwargs': {'enable_thinking': False}}, timeout=args.request_timeout)
            ids = tokens.get('tokens') if isinstance(tokens, dict) else None
            if not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids):
                raise ValueError('tokenizer did not return nonempty integer token IDs')
            record['input_token_count'] = len(ids)
            command(['docker', 'exec', args.container, 'mkdir', '-p', remote], timeout=15)
            before = snapshot(args.container, command)
            payload = {'profile_id': profile_id, 'output_dir': remote,
                       'activities': ['CPU', 'GPU'], 'profile_by_stage': False,
                       'merge_profiles': True, 'profile_prefix': profile_id}
            record['start_payload'] = payload
            record['window_start_unix_s'] = time.time()
            attempted = True  # Even a lost start response may leave a live profiler.
            record['start_response'] = http(base + '/start_profile', payload, timeout=30)
            # Matches run_trace_v4.generate's native sampling parameters; stdlib
            # nonstream transport avoids requiring requests on the profiling host.
            generation = {'rid': rid, 'input_ids': ids, 'stream': False,
                          'sampling_params': {'temperature': 0, 'top_p': 1, 'top_k': -1,
                                              'max_new_tokens': args.max_new_tokens}}
            record['sampling_params'] = generation['sampling_params']
            started = time.monotonic()
            record['generation_status'] = 'in_progress'
            response = http(base + '/generate', generation, timeout=args.request_timeout)
            record['client_generate_elapsed_ms'] = (time.monotonic() - started) * 1000
            if not isinstance(response, dict) or not isinstance(response.get('meta_info'), dict):
                raise ValueError('native generate response missing meta_info')
            if response['meta_info'].get('id') not in (None, rid):
                raise ValueError('native response request identity mismatch')
            record['native_meta'] = response['meta_info']
            record['generation_status'] = 'success'
        except Exception as exc:
            record['errors'].append(str(exc))
            if record['generation_status'] == 'in_progress':
                record['generation_status'] = 'error_or_unknown'
        finally:
            if attempted:
                try:
                    record['stop_response'] = http(base + '/stop_profile', {'profile_id': profile_id}, timeout=args.stop_timeout)
                    record['profile_stopped'] = True
                except Exception as exc:
                    record['errors'].append('stop_profile failed: ' + str(exc))
                record['window_end_unix_s'] = time.time()
                after = snapshot(args.container, command)
                try:
                    # One isolated remote directory per request; bounded directory copy
                    # preserves TP worker filenames and does not truncate discovery output.
                    command(['docker', 'cp', args.container + ':' + remote, str(local)], timeout=args.copy_timeout)
                    paths = sorted(p for p in local.rglob('*') if p.is_file() and p.name.endswith(('.json', '.json.gz')))
                    record['artifact_paths'] = [str(p) for p in paths]
                    record['artifact_map'] = [{'request_id': source['request_id'], 'profile_id': profile_id,
                        'path': str(p), 'remote_path': remote + '/' + p.relative_to(local).as_posix()}
                        for p in paths]
                except Exception as exc:
                    record['errors'].append('artifact collection failed: ' + str(exc))
        record['cpu'] = cpu_delta(before, after)
        parsed = parse_traces(record['artifact_paths'])
        record['kernel_summary'] = parsed
        record['errors'].extend(parsed['errors'])
        if not record['errors'] and record['profile_stopped']:
            record['profile_status'] = 'success' if record['cpu']['status'] == 'success' else 'partial'
        # Persist partial results after each window; never start a later window
        # after a failure or uncertain profiler/generation state.
        completed = len(result['requests']) == len(selected)
        statuses = [r['profile_status'] for r in result['requests']]
        result['status'] = ('success' if completed and all(s == 'success' for s in statuses)
                            else 'partial' if completed and all(s in ('success', 'partial') for s in statuses)
                            else 'error')
        summary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        if record['errors']:
            result['errors'].append('Stopped after request %s failed; remaining requests were not profiled.' % source['request_id'])
            break
    summary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return result, summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', required=True, choices=['dgx', '5880ada'])
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--requests', required=True, help='Existing final trace.jsonl containing real messages')
    parser.add_argument('--limit', type=int, default=2)
    parser.add_argument('--container', default='jiangchuan-qwen38-zh')
    parser.add_argument('--base-url', default='http://localhost:8000')
    parser.add_argument('--model', default='qwen3.8-27b-zh')
    parser.add_argument('--max-new-tokens', type=int, default=16)
    parser.add_argument('--request-timeout', type=float, default=120)
    parser.add_argument('--stop-timeout', type=float, default=120)
    parser.add_argument('--copy-timeout', type=float, default=90)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit < 1 or not 1 <= args.max_new_tokens <= 4096:
        parser.error('--limit must be positive and --max-new-tokens must be between 1 and 4096')
    if any(not 0 < value <= 600 for value in [args.request_timeout, args.stop_timeout, args.copy_timeout]):
        parser.error('timeouts must be positive and at most 600 seconds')
    try:
        result, path = collect(args)
    except Exception as exc:
        print(json.dumps({'status': 'error', 'error': str(exc)}))
        return 1
    print(json.dumps({'status': result['status'], 'summary': str(path),
                      'selected_count': result['selected_count'], 'profiled_count': len(result['requests'])}))
    return 0 if result['status'] == 'success' else 1


def load_requests(path, limit=2):
    if limit < 1:
        raise ValueError('limit must be positive')
    selected, seen = [], set()
    with Path(path).open(encoding='utf-8') as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            rid, messages = row.get('request_id'), row.get('messages')
            if not isinstance(rid, str) or not rid.strip() or rid in seen:
                raise ValueError('missing/duplicate request_id at line %d' % number)
            if not isinstance(messages, list) or not messages or any(
                    not isinstance(m, dict) or not isinstance(m.get('role'), str)
                    or 'content' not in m for m in messages):
                raise ValueError('missing/invalid messages at line %d' % number)
            selected.append({'request_id': rid, 'messages': messages, 'source_line': number})
            seen.add(rid)
            if len(selected) == limit:
                break
    if not selected:
        raise ValueError('no requests in trace')
    return selected


if __name__ == '__main__':
    raise SystemExit(main())
