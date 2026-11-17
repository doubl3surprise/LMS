"""
hardware.py — CPU/Memory/Disk-IO via psutil；GPU via pynvml；SGLang metrics
=========================================================================
两平台行为：
  5880ada：pynvml 在宿主机 venv 中可用（4× RTX 5880 Ada）；
           Docker 容器内无法访问 NVML，改用 sidecar 文件合并。
  DGX：pynvml 在 venv 中可用，但 GB10 统一内存显存字段返回 N/A；
       仅采集 GPU 利用率百分比。

v2 新增：
  - disk_read_mb / disk_write_mb（累计量，via psutil.disk_io_counters）
  - SGLang /metrics Prometheus 端点：kv_cache_tokens、queue_depth
"""

import re
import time
from dataclasses import dataclass, field

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import pynvml as _nvml
    _nvml.nvmlInit()
    _N_GPU = _nvml.nvmlDeviceGetCount()
    _HANDLES = [_nvml.nvmlDeviceGetHandleByIndex(i) for i in range(_N_GPU)]
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False
    _N_GPU = 0
    _HANDLES = []


@dataclass
class GpuSample:
    gpu_index: int
    util_pct: int | None          # GPU 核心利用率 %
    mem_used_mb: float | None     # 显存已用 MB（统一内存平台为 None）
    mem_total_mb: float | None


@dataclass
class HardwareSample:
    timestamp: float = field(default_factory=time.time)
    cpu_pct: float | None = None
    mem_used_gb: float | None = None
    mem_total_gb: float | None = None
    gpus: list[GpuSample] = field(default_factory=list)
    kv_cache_tokens: int | None = None    # 来自 SGLang /metrics
    queue_depth: int | None = None        # 来自 SGLang /metrics
    disk_read_mb: float | None = None     # 累计磁盘读取（MB）
    disk_write_mb: float | None = None    # 累计磁盘写入（MB）


def _fetch_sglang_metrics(base_url: str) -> dict:
    """
    轮询 SGLang /metrics Prometheus 端点，提取 KV cache tokens 和队列深度。
    超时 1.5s，失败静默返回空。

    常见 SGLang metric 名（版本间有差异，全部尝试）：
      kv cache:  sglang_gen_server_tokens_in_cache
                 sglang_kv_cache_usage_tokens
      queue:     sglang_num_waiting_reqs
                 sglang_waiting_requests
    """
    result = {"kv_cache_tokens": None, "queue_depth": None}
    try:
        import requests as _req
        resp = _req.get(f"{base_url}/metrics", timeout=1.5)
        if resp.status_code != 200:
            return result
        text = resp.text
        for pat in [
            r"sglang_gen_server_tokens_in_cache\s+([\d.]+)",
            r"sglang_kv_cache_usage_tokens\s+([\d.]+)",
            r'sglang_kv_cache_tokens_total\{[^}]*\}\s+([\d.]+)',
        ]:
            m = re.search(pat, text)
            if m:
                result["kv_cache_tokens"] = int(float(m.group(1)))
                break
        for pat in [
            r"sglang_num_waiting_reqs\s+([\d.]+)",
            r"sglang_waiting_requests\s+([\d.]+)",
        ]:
            m = re.search(pat, text)
            if m:
                result["queue_depth"] = int(float(m.group(1)))
                break
    except Exception:
        pass
    return result


def snapshot(sglang_url: str | None = None) -> HardwareSample:
    """
    采集一次当前硬件状态快照。
    在每道题 LLM 生成前后各调用一次，观察资源变化。

    sglang_url: 若提供则额外轮询 /metrics 获取 KV cache / 队列信息。
    """
    s = HardwareSample(timestamp=time.time())

    # ── CPU / 内存（psutil） ──────────────────────────────────────────
    if _HAS_PSUTIL:
        s.cpu_pct = psutil.cpu_percent(interval=0.05)
        vm = psutil.virtual_memory()
        s.mem_used_gb  = vm.used  / 1e9
        s.mem_total_gb = vm.total / 1e9

        # ── 磁盘 I/O（累计字节数） ─────────────────────────────────────
        try:
            io = psutil.disk_io_counters()
            if io:
                s.disk_read_mb  = io.read_bytes  / 1e6
                s.disk_write_mb = io.write_bytes / 1e6
        except Exception:
            pass

    # ── GPU（pynvml） ─────────────────────────────────────────────────
    if _HAS_NVML:
        for i, handle in enumerate(_HANDLES):
            g = GpuSample(gpu_index=i, util_pct=None, mem_used_mb=None, mem_total_mb=None)
            try:
                util = _nvml.nvmlDeviceGetUtilizationRates(handle)
                g.util_pct = util.gpu
            except Exception:
                pass
            try:
                mem = _nvml.nvmlDeviceGetMemoryInfo(handle)
                # GB10 统一内存下 mem.used 含义与独立显存不同，仍记录
                g.mem_used_mb  = mem.used  / 1e6
                g.mem_total_mb = mem.total / 1e6
            except Exception:
                pass
            s.gpus.append(g)

    # ── SGLang metrics（KV cache / queue depth） ──────────────────────
    if sglang_url:
        m = _fetch_sglang_metrics(sglang_url)
        s.kv_cache_tokens = m["kv_cache_tokens"]
        s.queue_depth     = m["queue_depth"]

    return s


def sample_to_dict(s: HardwareSample) -> dict:
    """把 HardwareSample 转为可 JSON 序列化的 dict。"""
    return {
        "timestamp":       s.timestamp,
        "cpu_pct":         s.cpu_pct,
        "mem_used_gb":     s.mem_used_gb,
        "mem_total_gb":    s.mem_total_gb,
        "gpus": [
            {
                "gpu_index":   g.gpu_index,
                "util_pct":    g.util_pct,
                "mem_used_mb": g.mem_used_mb,
                "mem_total_mb":g.mem_total_mb,
            }
            for g in s.gpus
        ],
        "kv_cache_tokens": s.kv_cache_tokens,
        "queue_depth":     s.queue_depth,
        "disk_read_mb":    s.disk_read_mb,
        "disk_write_mb":   s.disk_write_mb,
    }
