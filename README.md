# Math Reasoning Data Pipeline

用于从数学推理模型的多次生成结果中构建监督微调（SFT）数据的三阶段流水线，并提供可恢复的评测工具、独立 API 配置和本地 SGLang 服务管理。

## 功能概览

Pipeline 按顺序执行三个阶段：

1. **Stage 1：Rollout + Repetition Judge**
   - 每道题生成多条独立推理轨迹；
   - 判断是否出现连续、大量的机械重复；
   - 无复读轨迹直接成为 Stage 1 正向训练候选；
   - 有复读轨迹进入 Stage 2。
2. **Stage 2：Rerollout + Correctness Judge**
   - 对有复读的轨迹重新生成答案；
   - 使用独立 Judge API 判断预测答案与参考答案是否等价；
   - 正确轨迹成为 Stage 2 训练候选并进入 Stage 3。
3. **Stage 3：Reform + Correctness Judge**
   - 将 Stage 2 轨迹改写为干净、自包含的 SFT 样本；
   - 再次进行正确性判定；
   - 正确结果成为 Stage 3 训练候选。

项目还提供：

- Rollout API 与 Judge API 的物理配置隔离；
- 基于 JSONL checkpoint 的断点恢复；
- 单卡本地 SGLang 安全部署脚本；
- 两个对称的模型 benchmark 评测入口；
- 与 `model_iterations_correct` 一致的 compact training schema 说明。

## 目录结构

```text
repetition-code/
├── full_pipeline.sh              # Stage 1→2→3 统一入口
├── quickstart.sh                 # 默认数据和配置的快速验证入口
├── streaming_pipeline.py         # Stage 1 + Stage 2
├── turn3_reform.py               # Stage 3
├── share_utils.py                # API 客户端、prompt、boxed 和 Judge 解析工具
├── deploy_local_sglang.sh        # 只管理本次自建进程组的本地部署脚本
├── run_eval.py                   # 共享 benchmark 评测引擎
├── run_eval_model_a.sh           # Model A 评测入口
├── run_eval_model_b.sh           # Model B 评测入口
├── config/
│   ├── prompts/                  # 当前主 Pipeline 实际使用的 5 个 prompt
│   └── runtime/                  # Pipeline/API/Eval 配置及示例
├── data/
│   ├── README.md                 # 默认数据来源、字段和校验信息
│   ├── example_input.jsonl       # 一条真实的最小输入示例
│   └── pipeline_default_5k.jsonl # 本地默认 5K 数据，Git 忽略
├── pipeline_results/             # Pipeline 结果，Git 忽略
├── eval_results/                 # Benchmark 结果，Git 忽略
├── PIPELINE_OUTPUT_SCHEMA.md     # Stage 1/2/3 完整字段规范
└── DEMO_README.md                # 逐层数据示例和完整演示
```

## 安装

```bash
python -m pip install -r requirements.txt
```

依赖保持最小化：

- `openai`
- `httpx`
- `PyYAML`

## 配置

### 配置分层

```text
config/prompts/   可提交的 prompt 文本
config/runtime/   运行参数、endpoint、凭据和非敏感 example
```

当前主 Pipeline 只使用以下 5 个 prompt：

```text
config/prompts/rollout_prompt.yaml
config/prompts/judge_repeat_prompt_v4.yaml
config/prompts/judge_prompt.yaml
config/prompts/turn2_rerollout_v4.yaml
config/prompts/turn3_reform_v3.yaml
```

### 创建本地配置

```bash
cp config/runtime/pipeline.example.yaml config/runtime/pipeline.yaml
cp config/runtime/rollout_api.example.yaml config/runtime/rollout_api.yaml
cp config/runtime/judge_api.example.yaml config/runtime/judge_api.yaml
```

职责如下：

| 配置 | 内容 |
|---|---|
| `pipeline.yaml` | 输入字段、prompt 路径、采样参数、worker、重试和阶段参数 |
| `rollout_api.yaml` | `qwen_apex` 及可选的 `qwen_remote`，仅用于生成 |
| `judge_api.yaml` | `kimi`、`glm`、`deepseek_397b` 中至少一个，仅用于判定 |

真实 endpoint、API key、本地路径和运行配置均被 `.gitignore` 排除，不应提交。

> `rerollout.force_correct` 和 `turn3_reform.force_correct` 只能用于显式授权的端到端数据流验证。正常生产数据必须关闭这两个选项，并使用真实 Judge 结果。

## 默认输入数据

未传 `--dataset` 时，`full_pipeline.sh` 默认读取：

```text
data/pipeline_default_5k.jsonl
```

它来自本地提供的 Nemotron-SFT-Math-v3 reformed no-tools JSONL；绝对源路径有意不提交。

提取规则是按源文件顺序复制前 5,000 条有效 JSONL，不修改任何字段。文件约 217.7 MB，因此默认被 Git 忽略。

Pipeline 实际读取三个字段：

```json
{
  "uuid": "稳定样本 ID",
  "problem": "数学题目",
  "expected_answer": "参考答案"
}
```

可提交的一条真实最小示例位于：

```text
data/example_input.jsonl
```

完整来源、源字段和 SHA-256 见 [`data/README.md`](data/README.md)。

## Quickstart

### 只检查配置，不调用 API

```bash
bash quickstart.sh
```

默认行为是 dry-run：

- 校验默认 5K 数据；
- 校验 Pipeline、Rollout API、Judge API 和所有 prompt；
- 打印将执行的 Stage 1/2/3 命令；
- 不创建结果目录；
- 不发起网络请求。

### 实际执行

确认 endpoint 可用后：

```bash
bash quickstart.sh \
  --run \
  --limit 30 \
  --output-dir pipeline_results/demo_30
```

或直接使用统一入口：

```bash
bash full_pipeline.sh \
  --config config/runtime/pipeline.yaml \
  --rollout-api-config config/runtime/rollout_api.yaml \
  --judge-api-config config/runtime/judge_api.yaml \
  --dataset data/pipeline_default_5k.jsonl \
  --limit 30 \
  --output-dir pipeline_results/demo_30 \
  --run-id demo_30
```

不指定 `--output-dir` 时，默认写入：

```text
pipeline_results/YYYYmmdd_HHMM/
```

## 数据流

```text
input JSONL
  │
  ├─ Stage 1: rollout + repetition judge
  │    ├─ stage1/trajectory_raw.jsonl
  │    ├─ stage1/no_repetition.jsonl        → Stage 1 训练候选
  │    └─ stage1/has_repetition.jsonl       → Stage 2 输入
  │
  ├─ Stage 2: rerollout + correctness judge
  │    ├─ stage2/repetition_correct.jsonl   → Stage 2 候选 / Stage 3 输入
  │    └─ stage2/repetition_incorrect.jsonl
  │
  └─ Stage 3: reform + correctness judge
       ├─ stage3/turn3_reform_correct.jsonl → Stage 3 训练候选
       └─ stage3/turn3_reform_incorrect.jsonl
```

## Pipeline 输出

```text
pipeline_results/<run-id>/
├── full_pipeline.log
├── .pipeline.lock
├── service/                       # 仅本地部署时存在
│   ├── runtime.json
│   └── sglang.log
├── stage1/
│   ├── trajectory_raw.jsonl
│   ├── no_repetition.jsonl
│   └── has_repetition.jsonl
├── stage2/
│   ├── repetition_correct.jsonl
│   └── repetition_incorrect.jsonl
└── stage3/
    ├── turn3_reform_correct.jsonl
    └── turn3_reform_incorrect.jsonl
```

### Stage 1

- `trajectory_raw.jsonl`：每次成功 rollout 的原始审计轨迹，不包含 Judge 结论；
- `no_repetition.jsonl`：未达到复读阈值的正向训练候选；
- `has_repetition.jsonl`：达到阈值的代表轨迹，进入 Stage 2。

### Stage 2

- `repetition_correct.jsonl`：rerollout 后判对，包含 `final_messages`；
- `repetition_incorrect.jsonl`：所有允许尝试均未通过的终态审计数据。

### Stage 3

- `turn3_reform_correct.jsonl`：reform 后判对，新的 `final_messages` 是 Stage 3 训练轨迹；
- `turn3_reform_incorrect.jsonl`：能解析但未判对的审计记录。

所有顶层字段、Judge 子字段、attempt 字段和 messages 结构见：

- [`PIPELINE_OUTPUT_SCHEMA.md`](PIPELINE_OUTPUT_SCHEMA.md)
- [`DEMO_README.md`](DEMO_README.md)

## 断点恢复

各阶段输出均为 append-only JSONL checkpoint。

- Stage 1/2 使用 `uuid` 集合识别已分类和终态记录；
- Stage 3 使用 `--skip-existing` 跳过已完成 UUID；
- 同一 `--output-dir` 可重复运行；
- `--limit N` 现在会读取满 N 条即停止，不再先加载整个超大 JSONL。

同一个输出目录由 `.pipeline.lock` 防止并发写入。

## Compact Training Data

Pipeline 的 Stage 文件是包含 Judge 和重试信息的审计数据，不与最终训练数据完全相同。

参考 `model_iterations_correct` 的最终训练记录严格只有三个顶层字段：

```json
{
  "uuid": "problem identifier",
  "messages": [
    {"role": "user", "content": "problem or rendered prompt"},
    {
      "role": "assistant",
      "reasoning_content": "reasoning",
      "content": "final response"
    }
  ],
  "source_stage": "stage1"
}
```

转换关系：

| 训练阶段 | Pipeline 来源 | 训练 `messages` |
|---|---|---|
| Stage 1 | `stage1/no_repetition.jsonl` | `messages` |
| Stage 2 | `stage2/repetition_correct.jsonl` | `final_messages` |
| Stage 3 | `stage3/turn3_reform_correct.jsonl` | `final_messages` |

转换时只保留：

```text
uuid
messages
source_stage
```

Judge 证据、参考答案、重试记录和中间历史不会进入 compact training record。

## 本地 SGLang 部署

启动一个显式 GPU 上的本地服务：

```bash
bash deploy_local_sglang.sh start \
  --model-path /path/to/Qwen3.5-4B \
  --gpu 0 \
  --port 8000 \
  --run-dir pipeline_results/demo_30
```

然后将 `config/runtime/rollout_api.yaml` 指向：

```text
http://127.0.0.1:8000/v1
```

停止本次服务：

```bash
bash deploy_local_sglang.sh stop \
  --run-dir pipeline_results/demo_30
```

安全约束：

- 端口占用时直接失败，不清理占用者；
- 不使用全局 `pkill`、`pgrep` 或 `fuser -k`；
- 不扫描或停止训练进程；
- 停止前校验 PID、PGID、进程 starttime、模型路径和端口；
- 只管理该 run 目录记录的进程组。

## 双模型 Benchmark 评测

评测脚本接收已经启动的 OpenAI-compatible API URL，不负责部署或停止模型。

### Benchmark 配置

```bash
cp config/runtime/eval_benchmarks.example.yaml \
   config/runtime/eval_benchmarks.yaml
```

### Model A

Model A 默认直接读取：

```text
config/runtime/judge_api.yaml
```

默认 Judge backend 为 `deepseek_397b`，因此通常只需传 benchmark 配置和采样超参：

```bash
bash run_eval_model_a.sh http://model-a:8001/v1 default all \
  --comparison-id experiment_001 \
  --benchmark-config config/runtime/eval_benchmarks.yaml \
  --passk 8 \
  --temperature 0.6 \
  --max-concurrent 40
```

如需切换 Judge，仍可显式传入 `--judge-api-config` 和 `--judge-backend` 覆盖默认值。

### Model B

Model B 默认读取独立超参文件：

```text
config/runtime/eval_model_b.yaml
```

创建本地配置：

```bash
cp config/runtime/eval_model_b.example.yaml \
   config/runtime/eval_model_b.yaml
```

运行：

```bash
bash run_eval_model_b.sh http://model-b:8001/v1 default all
```

CLI 参数仍可覆盖配置值，也可显式切换配置：

```bash
bash run_eval_model_b.sh http://model-b:8001/v1 default all \
  --eval-config config/runtime/eval_model_b.example.yaml
```

评测输出：

```text
eval_results/<comparison_id>/
├── model_a/
│   ├── manifest.json
│   ├── logs/<benchmark>.log
│   └── <benchmark>/run_<k>/
│       ├── traj_<index>.json
│       └── brief.json
└── model_b/
    └── ...同一结构...
```

`pass_k` 表示目标总 run 数。重复执行会跳过有效 trajectory 并补齐缺失任务。

## 已验证结果

本项目完成过两类 30 UUID 验证：

1. **自然分流**
   - Stage 1 raw：60 条；
   - Stage 1 no-repetition：60 条；
   - 30 个 UUID 均未进入 Stage 2，Stage 3 按真实路由跳过。
2. **显式授权的强制端到端验证**
   - Stage 1 has-repetition：30 条；
   - Stage 2 correct：30 条；
   - Stage 3 correct：30 条；
   - 因 Judge endpoint 配额耗尽，Stage 2/3 明确包含 `forced_correct` 和 `force_reason`，只用于验证数据流和 schema，不应作为真实正确性标签。

详细实测表见 [`PIPELINE_OUTPUT_SCHEMA.md`](PIPELINE_OUTPUT_SCHEMA.md)。

## 验证命令

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
bash -n full_pipeline.sh
bash -n quickstart.sh
bash quickstart.sh
```

当前单元测试覆盖 benchmark 加载、字段映射、输出路径、trajectory 验证、brief 汇总和 URL 脱敏。

## 安全与仓库策略

以下内容默认不会进入 Git：

- 实际 endpoint/API key；
- `config/runtime/*.yaml` 本地运行配置；
- 默认 5K 数据文件；
- `pipeline_results/`；
- `eval_results/`；
- 日志、缓存和运行环境。

可以提交：

- prompt；
- `*.example.yaml`；
- `data/README.md`；
- `data/example_input.jsonl`；
- 代码、测试和 schema 文档。

## 更多文档

- [`DEMO_README.md`](DEMO_README.md)：逐层数据流、JSON 示例和完整命令
- [`PIPELINE_OUTPUT_SCHEMA.md`](PIPELINE_OUTPUT_SCHEMA.md)：Stage 1/2/3 字段规范和实测统计
- [`data/README.md`](data/README.md)：默认数据来源、字段和校验值
- [`WORK_LOG.md`](WORK_LOG.md)：分钟级实施与验证记录
