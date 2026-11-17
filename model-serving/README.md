# 本地 Qwen3.8-27B 中文助手

部署目录：`/home/jiangchuan/mem0-reproduction/model-serving`。使用 Qwen 官方 BF16 原始权重和本机已有的 SGLang Docker 镜像，通过 GB10 运行。下载完成后约占 56GB 模型存储空间；Docker 镜像占用另计。服务绑定 `127.0.0.1:8000`，模型名为 `qwen3.8-27b-zh`。

## 查看部署进度

```bash
cd /home/jiangchuan/mem0-reproduction/model-serving
./status.sh
systemctl --user status qwen38-setup.service
tail -n 40 logs/model-download.log
```

首次安装任务先下载、逐个校验 SHA-256，再启动模型并进行四次实际推理。`logs/service-verification.log` 记录推理验证结果，`logs/smoke-test.json` 保存测试回答；这两个文件只有运行到验证阶段才会生成。配置检查通过不代表权重已下载或模型已成功运行。

下载使用 Qwen 官方魔搭与 Hugging Face 仓库；18 个权重分片在两处的 SHA-256 已核对一致。文件版本、大小和 SHA-256 保存在 `downloads/modelscope-files.json` 和 `downloads/huggingface-files.json`。下载支持断点续传，所有文件验证通过后才生成 `models/Qwen3.8-27B/download-verified.json`。已有文件不会因为重新执行安装而重复下载。下载中断后运行：

```bash
./setup.sh
```

不要同时启动多个安装任务。脚本使用文件锁避免冲突。当前后台安装由用户级 systemd 管理；退出所有登录会话可能停止尚未完成的安装。已启动的模型由 Docker 的重启策略管理。

## 对话

服务健康检查成功后：

```bash
./chat.sh
./chat.sh "请用三句话解释什么是缓存。"
printf '%s\n' '请翻译成中文：This feature reduces memory usage.' | ./chat.sh
./chat.sh --think --max-tokens 8192 "分析这个问题的条件和可能结论……"
```

交互模式支持 `/clear` 清空上下文和 `/exit` 退出。在支持粘贴保护的终端中，直接粘贴多行文字后按回车，整段会作为一次提问发送。也可以先输入 `/paste`，逐行输入或粘贴，最后单独一行输入 `/send` 发送，或 `/cancel` 取消：

```text
你：/paste
… 请将下面的文字整理成连贯的段落：
… 第一行内容
… 第二行内容
… /send
助手：……
```

从文件提问可以使用 `./chat.sh < question.txt`，文件中的换行会保留。默认关闭思考模式，使用适合直接回答的采样参数。会话只保存在当前客户端进程内。超出上下文限制后请清空会话，或调整服务配置。

空回复会显示服务返回的 `finish_reason` 和生成 token 数（服务提供时）。只有 `length` 表示输出额度耗尽；若为 `stop`，先确认已发送完整问题，必要时 `/clear` 后重试。响应流中断会单独报错，不会把不完整的回复写入会话。普通的长度截断仍显示已生成的正文并提示增大 `--max-tokens`。

也可以使用聊天 API：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b-zh",
    "messages": [{"role":"user","content":"请解释什么是缓存。"}],
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "presence_penalty": 1.5,
    "max_tokens": 1024
  }'
```

API 基地址：`http://127.0.0.1:8000/v1`。本地服务未配置 API 密钥；若客户端要求填写，可填 `local`。API 调用方可以通过 `chat_template_kwargs: {"enable_thinking": true}` 启用思考。未显式传入的采样参数使用 SGLang 默认值。

## 通过 Pi Agent 使用

Pi 已配置为默认连接本机服务。在需要处理的项目目录中启动：

```bash
cd /你的项目目录
pi
```

也可以直接提问、继续上一次会话，或显式指定本地模型：

```bash
pi -p "请用中文解释什么是缓存。"
pi --continue
pi --model qwen-local/qwen3.8-27b-zh
pi --offline
```

`--offline` 禁用 Pi 启动时的联网操作，仍可访问本机模型服务；它不限制模型通过工具执行的命令。模型推理地址是 `http://127.0.0.1:8000/v1`，无须登录云端或购买 API。Pi 在项目目录中默认可使用 `read`、`bash`、`edit`、`write` 工具；只聊天时可运行 `pi --no-tools`。

默认关闭思考，以便直接回答。在 Pi 内使用 `/thinking` 切换，或启动时指定：

```bash
pi --thinking low
pi --thinking medium
pi --thinking high
```

Pi 的 `minimal`、`low` 对应模型的 `low`，`medium` 对应 `medium`，`high`、`xhigh`、`max` 对应 `xhigh`。思考与正文共用每次最多 8192 tokens 的输出额度。服务上下文是 32768 tokens，Pi 已为该模型单独设置自动压缩参数。

模型配置在 `~/.pi/agent/models.json`，默认模型和压缩设置在 `~/.pi/agent/settings.json`。修改前的设置已备份为同目录的 `settings.json.bak-20261005-194117`。模型沿用本服务已有的中文表达规则。

如果 Pi 已在配置前启动，退出后重新运行 `pi`；或者通过 `/model` 选择 `qwen-local/qwen3.8-27b-zh`。旧会话可能恢复原来的模型，必要时显式传入 `--model qwen-local/qwen3.8-27b-zh`。

出现连接错误时先检查服务：

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
/home/jiangchuan/mem0-reproduction/model-serving/status.sh
```

若模型容器已停止，运行 `/home/jiangchuan/mem0-reproduction/model-serving/start.sh`，等待健康检查成功后再启动 Pi。该启动脚本会重新加载模型，健康运行时无须重复执行。

本机已实测中文回复、思考模式、文件读取与修改、新建文件和执行验证脚本，结果保存在 `logs/pi-agent-verification.json`，详细事件记录在同目录的 `pi-agent-*.jsonl`。Pi 的自定义模型配置方式参见[官方文档](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/models.md)。

## 中文 skill

规则位于 `skills/natural-chinese/SKILL.md`，要求自然准确的现代汉语、具体的用词和直接的说明，禁用比喻、拟人和类比，减少生硬翻译、套话和重复结论。翻译、改写和润色默认只输出修改后的正文；保留事实与不确定程度，将修辞改为实际含义；明确要求逐字引用的引文保持原样。

`scripts/build_template.py` 把 skill 正文嵌入官方聊天模板。通过本服务的 `/v1/chat/completions` 调用时，规则自动进入系统消息，无须手动重复粘贴；已有的系统消息、多轮对话、工具描述和思考开关均保留。原始权重和官方模板文件不作修改。直接使用 `/generate` 或文本补全接口不会自动应用这份聊天模板。

这是提示词约束，没有进行微调，也不保证每次输出都符合文风。可以查看 `logs/smoke-test.json` 中的真实回答，继续调整规则。修改后执行：

```bash
./start.sh
```

该命令会重新生成模板并重建本部署的容器，使新规则生效。模型重新加载需要时间。

## 管理

```bash
./start.sh                         # 启动或重新加载
./stop.sh                          # 停止模型，释放占用
docker compose logs --tail=100 -f  # 查看启动和推理日志
```

默认上下文为 32768 tokens，当前有效推理并发为 1，其余请求会排队；静态内存分配比例设为 0.75，为临时计算和主机保留余量。模型本身支持的更长上下文未在本部署中启用。权重下载完成后的第一次服务启动，包含加载和预热，本机实测约 8 分半；应以 `/health` 返回成功为准。

首次加载前，脚本检查可用主存至少为 85GiB；GB10 使用统一内存，这能为加载和临时计算保留空间。若其他任务占用较多内存，脚本会提示稍后重试。

运行时启用 Hugging Face 离线模式，推理使用本地权重。镜像使用已有的 `lmsysorg/sglang:qwen38-27b`，本次检查到的镜像 ID 保存在 `runtime-image-id.txt`，不会自动拉取其他版本。

官方资料：[Qwen 模型说明](https://huggingface.co/Qwen/Qwen3.8-27B)、[魔搭权重仓库](https://modelscope.cn/models/Qwen/Qwen3.8-27B)、[SGLang 部署文档](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)。
