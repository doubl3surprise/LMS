"""Generate the bounded Task4 Markdown report from verified outputs."""
import json

def write_report(result,curves,out):
 t='''# 任务四：长期记忆访问模式与缓存复用机会

## 范围与计量

使用现有真实 LoCoMo 问题驱动的已测 Memory Access Trace，不生成合成性能数据、不启动新推理、不修改记忆库。各批次按原有执行顺序独立回放，冷缓存起步，不拼接批次放大热点。只模拟最终选中记忆对象的读取：未记录的候选读取、向量索引页读取不在范围内。

原生 Mem0 与自定义 MMR 分开统计；两个数据集题目数量/输出策略不同，不能将热点差异单独归因于选择策略。32题与152题均来自 conv-26，同一原始conversation不是多个实际访问会话。9轮是人工编排的3个真实访问会话，含实际前轮问答，不代表自然用户的会话分布。

对象身份=(user/数据集隔离命名空间,Memory ID)，文本身份=原始UTF-8字节SHA256。不做同义/相似文本归并。发现同ID文本变化即拒绝分析，防止把旧版本误判可复用。文本容量仅正文UTF-8，不含向量、metadata、Python对象开销，也不是KV字节。

## 访问频率、热点与跨请求重复

|平台|批次|请求|对象访问|唯一ID|唯一文本|跨请求重复ID访问|Top10对象访问占比|有跨会话访问的对象|
|---|---|---:|---:|---:|---:|---:|---:|---:|
'''
 for host,groups in result['cohorts'].items():
  for tag,s in groups.items():
   a=s['objects'];b=s['texts'];cross=a['cross_session_objects']
   t+=f"|{host}|{tag}|{s['requests']}|{a['accesses']}|{a['unique']}|{b['unique']}|{a['prior_request_repeat_accesses']}|{a['top10_share']:.1%}|{cross if cross is not None else 'N/A：无有效访问会话'}|\n"
 t+='\n跨请求重复统计不含同一请求内部的重复；频率与模拟按逐对象访问计数。跨会话只读实际(user_id,request_session_id)，不使用答案evidence、原始记忆来源会话或replay批次号。\n\n'
 for tag in ['native32','mmr152','sessions']:
  s=result['cohorts']['dgx'][tag];rd=s['objects']['reuse_distance']
  t+=f"### {tag}\n\n正文工作集 {s['utf8_unique_object_bytes']} bytes；冷访问 {rd['cold']}；非冷重用距离中位数 {rd['median_non_cold']}。\n\n![对象频率]({tag}_objects_frequency.svg)\n\n![文本频率]({tag}_texts_frequency.svg)\n\n![重用距离]({tag}_reuse.svg)\n\n"
 t+='''## 容量敏感性：冷启动 LRU / LFU

重用距离=同一对象两次访问之间的不同对象数，即LRU栈距离；首次访问为cold，不记0。用不变正文的顺序需求访问模拟。LRU按最近访问淘汰；LFU按驻留期间计数淘汰、同频按最久未用，重新装入重置计数。无未来信息，无预取，超大对象绕过且不清空缓存。每条LRU对象容量结果均与“距离<容量”独立恒等式校验。

|批次|容量对象数|LRU命中率|LFU命中率|
|---|---:|---:|---:|
'''
 for tag in ['native32','mmr152','sessions']:
  for cap in [10,32,64,128]:
   matches={r['policy']:r['hit_rate'] for r in curves if r['host']=='dgx' and r['cohort']==tag and r['identity']=='object_key' and r['unit']=='objects' and r['capacity']==cap}
   t+=f"|{tag}|{cap}|{matches['LRU']:.1%}|{matches['LFU']:.1%}|\n"
  t+=f"\n![对象容量]({tag}_objects_capacity.svg)\n\n![正文容量]({tag}_bytes_capacity.svg)\n\n"
 t+='''完整的对象/文本身份、逐对象访问命中比例和字节加权命中比例见[容量CSV](capacity_curves.csv)。图中容量不等距，横轴使用实际线性容量。LFU不保证胜过LRU，短批次热身和请求次序影响曲线；这些是本轨迹结果，不是生产命中率预测。

## 对象重复、文本重复与KV复用

- 对象命中允许复用读出的正文；相同文本可跨不同ID去重。两者都不自动允许复用该片段的KV。
- 因果注意力中，片段的KV依赖之前完整token前缀、位置与模型状态。相同记忆排在不同位置、前面的问题或记忆顺序变化，都可能使原KV不兼容。混合Mamba还需兼容状态；本报告不套纯Transformer KV字节公式。
- 总前缀机会使用实际token序列、只比较先前请求、假设无限驻留的最长共同前缀；是逻辑机会的乐观上界。模板相同也会贡献前缀，不能把全部前缀token算成记忆KV复用。并发/淘汰、块边界、缓存命名空间、Mamba状态会进一步约束实际命中。
- 质量Trace在system开头含每请求不同marker，阻断后续记忆前缀复用。不能从其零命中推出生产访问没有缓存价值。9轮会话需另核对双机实际生成历史；本批次导出的完整Prompt token哈希逐条一致，不能仅凭Memory ID不同推断上下文不同。

|平台|批次|实测cached tokens总数|Prompt tokens总数|
|---|---|---:|---:|
'''
 for host,groups in result['cohorts'].items():
  for tag,s in groups.items():t+=f"|{host}|{tag}|{s['observed_cached_tokens']}|{s['prompt_tokens']}|\n"
 t+='\n|平台|批次|精确token前缀逻辑机会|占Prompt tokens|会话内机会（不跨会话）|\n|---|---|---:|---:|---:|\n'
 for host,groups in result['cohorts'].items():
  for tag,s in groups.items():
   p=s['prefix']
   if p:
    t+=f"|{host}|{tag}|{p['logical_repeated_tokens']}|{p['logical_repeated_tokens']/p['prompt_tokens']:.2%}|{p.get('within_session_repeated_tokens','N/A')}|\n"
 t+='\n假设块粒度16/64/256 token的向下取整机会单独保存在JSON的hypothetical_block_tokens，不代表引擎实际配置。\n\n'
 for tag in ['native32','mmr152','sessions']:t+=f'![token级Prefix机会](dgx_{tag}_prefix.svg)\n\n'
 t+='\n### 记忆正文与兼容前缀的交集\n\n|平台|批次|核验请求/总数|正文token访问|先前请求出现过的正文token|其中兼容前缀token|兼容比例|实测cached前缀投影交集|\n|---|---|---:|---:|---:|---:|---:|---:|\n'
 for host,groups in result['cohorts'].items():
  for tag,s in groups.items():
   m=s['memory_prefix'];c=m['counts'];den=c['previously_seen_text_tokens'];rate=c['compatible_repeated_text_tokens']/den if den else 0
   t+=f"|{host}|{tag}|{m['verified_requests']}/{s['requests']}|{c['memory_tokens']}|{den}|{c['compatible_repeated_text_tokens']}|{rate:.2%}|{c['cached_prefix_repeated_text_tokens']}|\n"
 t+='\n正文范围由实际token ID解码后再编码完全相同、最后user消息的唯一正文或[Memory n]标签定位得到；token边界按与正文字符相交计算，排除历史轮次和标签主体。重复正文指先前请求出现过相同完整UTF8文本，部分词语相同不算完整正文重复。兼容比例是无限驻留、块大小1的输入前缀机会，非实测命中率。cached前缀投影按引擎cached_tokens是连续前缀长度的语义求交，不是逐token缓存事件；不同批次、Judge和外部请求可影响驻留，不能将其与当前轨迹的机会无条件相除。\n\n'
 t+='\nPrefix逻辑机会逐批次指标见[结构化摘要](summary.json)中的prefix；未取得精确token序列时为null，不用字符长度估算。实测cached_tokens可受同批次以外请求和Judge影响；无限历史上界只包含本批次已见输入，二者不能不加限定直接相除为“系统覆盖率”。本报告尚未把重复记忆的所有token逐一归因到实测命中，不能报告精确“记忆KV覆盖百分比”。\n\n'
 t+='''### 相同完整Prompt立即重放：已有实测对照（LLM-only）

|平台|路径|配对数|repeat命中请求|first Prefill ms|repeat Prefill ms|repeat平均cached tokens|
|---|---|---:|---:|---:|---:|---:|
'''
 for host,groups in result['replay'].items():
  for c,s in groups.items():t+=f"|{host}|{c}|{s['pair_count']}|{s['repeat_hit_requests']}|{s['prefill_first_mean_ms']:.1f}|{s['prefill_repeat_mean_ms']:.1f}|{s['repeat_cached_tokens_mean']:.1f}|\n"
 t+='''
这组证明现有Prefix机制在兼容完整前缀下可复用计算，但只覆盖相同Prompt立即重放；不代表记忆对象缓存命中率，也不代表不同检索结果可以任意拼接KV。first未清空全局缓存，不是系统冷启动。

## 跨平台容量与访问成本

同样访问序列、缓存单位和容量下，LRU/LFU逻辑结果与硬件无关。分别保存的双机序列一致性如下；会话组可能受生成答案影响，但比较对象顺序与输入token是不同检查。

'''
 for tag,x in result['cross_platform'].items():t+=f"- {tag}：问题及Memory ID/文本顺序一致={x['same_question_and_id_text_order']}；只比较问题及正文顺序一致={x['same_question_and_text_order']}。不同ID可指向完全相同文本，正文一致时不归为硬件导致的内容变化。\n"
 for host,h in result['hardware'].items():
  if h:
   info=json.loads(h['discovery_stdout']);t+=f"\n{host}：`{info['mem_total_kib']}`；GPU查询：`{h['gpu_stdout'].strip().replace(chr(10),'; ')}`。\n"
 t+='\n### 正文字节预算与传输量敏感性（模拟）\n\n|平台|批次|正文预算bytes|策略|对象命中率|字节加权命中率|省去正文读取bytes|\n|---|---|---:|---|---:|---:|---:|\n'
 for row in curves:
  if row['identity']=='object_key' and row['unit']=='bytes' and row['capacity'] in [4096,16384,65536] and row['policy']=='LRU':t+=f"|{row['host']}|{row['cohort']}|{row['capacity']}|LRU|{row['hit_rate']:.1%}|{row['byte_hit_rate']:.1%}|{row['hit_bytes']}|\n"
 t+='''
GB10共享系统内存，5880Ada是分离显存；4张显存不能直接当作单卡可用池相加。采集到的总容量不是可分配缓存预算，模型权重、引擎预分配及Mamba状态会占用空间。正文对象工作集很小，不能据此推断完整879条索引或GPU KV池的容量压力。

成本敏感性采用明确的模拟形式：给定逐次miss成本 c_m、hit成本 c_h，收益=(原miss数−模拟miss数)×(c_m−c_h)；或传输部分约为hit_bytes/B。这里的B必须是相应存储/链路的有效带宽，不是GPU峰值带宽。本轮未测正文随机读取的miss/hit成本，也没有两平台绝对GB/s证据，因此不填写伪造的毫秒收益。SDK检索总延迟还含Embedding与搜索，不能按对象命中率整体扣减。现有Prefix重放的Prefill实测差值独立列出。

## 复现与限制

在项目task4目录执行：

```bash
python3 -m unittest discover -p 'test_*.py' -v
python3 analyze.py --inputs inputs --out ../trace_results/task4
```

输出为SVG、Markdown、CSV和JSON，不需要GPU、不发起推理。保留原始Trace快照及哈希。范围覆盖局部性/频率、对象与字节容量模拟、明确边界的KV机会与已有实测复用；仍缺更大原生访问样本、自然多会话顺序、逐记忆KV命中归属和正文缓存成本微基准。这里不实现新KV管理算法，也不宣称验证了任意片段KV重组。
'''
 (out/'report.md').write_text(t)
