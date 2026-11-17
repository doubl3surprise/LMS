"""
config.py — 平台配置（路径、URL、参数）
========================================
同一份代码在两台机器上通过 --platform 参数选择配置。
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlatformConfig:
    platform: str
    sglang_url: str         # http://127.0.0.1:8000
    qdrant_mode: str        # "server" | "file"
    qdrant_url: str | None  # server 模式
    qdrant_path: str | None # file 模式
    dataset_path: str
    trace_results_dir: str  # run_id 目录的父目录
    model: str = "qwen3.8-27b-zh"
    user_id: str = "locomo_0_temporal_fix"
    collection: str = "memories"
    cand_k: int = 50
    final_k: int = 10
    mmr_lambda: float = 0.5
    # chat template 固定开销（token 数）：
    #   5880ada 用标准 Qwen3 chat_template.jinja → ~18 特殊 token
    #   DGX 与 5880ada 使用相同标准模板 → ~18 token（近似）
    #   这个值通过发送最小 prompt 测量得出（见 run_trace.py --calibrate-overhead）
    template_overhead_tokens: int = 18


PLATFORM_CONFIGS: dict[str, PlatformConfig] = {
    "5880ada": PlatformConfig(
        platform="5880ada",
        sglang_url="http://127.0.0.1:8000",
        qdrant_mode="server",
        qdrant_url="http://127.0.0.1:6333",
        qdrant_path=None,
        dataset_path="/workspace/datasets/locomo/locomo10.json",
        trace_results_dir="/workspace/trace_results",
        template_overhead_tokens=18,    # 标准 Qwen3 chat template
    ),
    "dgx": PlatformConfig(
        platform="dgx",
        sglang_url="http://127.0.0.1:8000",
        qdrant_mode="file",
        qdrant_url=None,
        qdrant_path="/home/jiangchuan/mem0-reproduction/benchmark-data/qdrant_imported",
        dataset_path="/home/jiangchuan/mem0-reproduction/datasets/locomo/locomo10.json",
        trace_results_dir="/home/jiangchuan/mem0-reproduction/trace_results",
        template_overhead_tokens=18,    # unified standard template; still approximate
    ),
}


def get_config(platform: str) -> PlatformConfig:
    if platform not in PLATFORM_CONFIGS:
        raise ValueError(f"Unknown platform: {platform!r}. Choices: {list(PLATFORM_CONFIGS)}")
    return PLATFORM_CONFIGS[platform]
