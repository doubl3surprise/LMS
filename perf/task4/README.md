# Task4：Memory Access Trace 驱动的缓存分析

主入口：`python3 analyze.py --inputs inputs --out ../trace_results/task4`。
测试：`python3 -m unittest discover -p 'test_*.py' -v`。

不需要模型服务或第三方Python包即可离线重新生成结果。实际token ID已从两台服务只读tokenize导出，每条长度及SHA256与历史Trace核对。当前inputs/native32文件每台96条，主分析显式过滤32条native_n879_k10，不把完整历史或无记忆混入对象缓存模拟。MMR152和3段9轮会话分别统计。

`trace_adapter.py`：对象和原文标识、有效访问会话及重用计数。
`cache_core.py`：精确不同对象重用距离、LRU/LFU对象/字节容量顺序模拟。
`prefix_core.py`：实际token前缀的历史匹配机会，非实测KV命中。
`analyze.py`：批次核验、模拟、CSV/JSON和图表生成。
`report.py`：生成中文方法与结果报告。
`figures.py`：标准库SVG生成。

原始输入哈希在inputs/manifest.json；实际分析的来源哈希写入输出summary.json。正文UTF8容量不是KV字节或索引总内存。批次为真实问题驱动的小样本访问，不是自然用户行为日志。不得将模拟命中率整体乘SDK检索时延当成实际加速。
