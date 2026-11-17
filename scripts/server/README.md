# scripts/server/ — 服务启动脚本

## start_mem0.sh

启动 Mem0 OSS 本地 HTTP 适配器服务（监听 127.0.0.1:8888）。

**用法**:
```bash
scripts/server/start_mem0.sh
```

**输出**:
```
[start_mem0] 启动 Mem0 适配器服务...
[start_mem0] 数据目录: .../benchmark-data
[start_mem0] 监听地址: http://127.0.0.1:8888

INFO: Uvicorn running on http://127.0.0.1:8888
```

**停止**: Ctrl+C

## 前置：启动 SGLang

**第一次**:
```bash
~/start_sglang_qwen38.sh
```

输出应该说 "Container ... already exists" 或 "... started"。

**检查**:
```bash
curl http://127.0.0.1:8000/v1/models
```

---

**最后更新**: 2026-11-17
