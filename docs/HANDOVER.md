# RadClaw 项目交接文档

## 1. 项目概述

RadClaw 是一个多智能体医学影像报告生成系统。给定一张医学影像（如胸部X光），系统通过 4 个 Agent 协作，自动生成结构化放射学报告。

核心特点：
- 4-Agent 流水线：Visual Analysis → Evidence Retrieval → Report Writing → Quality Review
- 质量反馈循环：QR Agent 可以驳回报告，触发定向重跑（最多 3 次迭代）
- 三层知识库：Terminology（RadLex/SNOMED）、Literature（远程指南 + PubMed）、Cases（远程案例）
- 26 个专用工具分布在 4 个 Agent 中
- 流式事件输出，支持实时追踪

---

## 2. 目录结构

```
radclaw/
├── config/
│   └── settings.yaml          # 主配置文件
├── data/
│   ├── knowledge/             # 本地知识库（RadLex、SNOMED）
│   │   └── terminology/       # RadLex 全集 JSON
│   ├── indexes/               # FAISS + BM25 本地索引缓存
│   └── raw/                   # 原始数据（RadLex.owl 等）
├── retrieval_service/         # 独立的远程检索服务（FastAPI）
│   ├── __init__.py
│   ├── app.py                 # FastAPI 应用入口
│   ├── config.py              # 服务配置 dataclass
│   ├── config_example.yaml    # 生产配置模板
│   ├── index.py               # 数据加载 + FAISS/BM25 索引构建
│   ├── search.py              # 检索 + RRF 融合
│   ├── run.py                 # 独立启动脚本
│   └── test_service.py        # 全量测试脚本
├── scripts/
│   ├── run_single.py          # 跑单条样本
│   ├── run_batch.py           # 批量跑样本
│   ├── run_trace.py           # 跑样本 + 保存完整 Agent 轨迹
│   ├── run_ablation.py        # 消融实验（B1/B2/B3/Full）
│   ├── evaluate.py            # 6 指标评估
│   ├── serve_model.sh         # 模型服务启动
│   ├── download_metric_models.py
│   └── ingest/
│       └── build_indexes.py   # 构建本地 FAISS 索引
├── src/
│   ├── core/
│   │   ├── orchestrator.py    # 主流水线编排
│   │   ├── loop.py            # 单 Agent 执行引擎（tool-use loop）
│   │   ├── baselines.py       # 消融基线（B1 direct, B2 visual_only, B3 retrieval_only）
│   │   ├── types.py           # 所有核心数据类型
│   │   ├── context.py         # 上下文管理
│   │   └── tool.py            # ToolRegistry 基类
│   ├── agents/
│   │   └── base.py            # Agent 工厂（build_agent）
│   ├── data/
│   │   ├── chexpert_loader.py # CheXpert-plus-RRG 数据加载
│   │   └── image_store.py     # 会话级图像存储
│   ├── hooks/
│   │   └── radiology_hooks.py # 质量钩子
│   ├── kb/
│   │   ├── embedder.py        # MedicalEmbedder（sentence-transformers）
│   │   ├── radlex_source.py   # RadLex FAISS 检索
│   │   ├── snomed_source.py   # SNOMED 检索
│   │   ├── guidelines_store.py# 本地指南存储（已被远程替代）
│   │   └── pubmed_client.py   # PubMed API 客户端
│   ├── llm/
│   │   └── client.py          # LLM 客户端（OpenAI 兼容 API）
│   └── tools/
│       ├── visual/            # 8 个视觉分析工具
│       ├── evidence/          # 7 个证据检索工具
│       ├── report/            # 5 个报告生成工具
│       └── review/            # 6 个质量审查工具
├── serve_sglang.sh            # SGLang 模型服务启动脚本
├── pyproject.toml             # Python 项目配置
└── output/                    # 运行输出目录
```

---

## 3. 环境依赖

### 3.1 Python 版本

Python >= 3.10

### 3.2 核心依赖

```bash
pip install -e .
```

`pyproject.toml` 中的核心依赖：

| 包 | 用途 |
|---|---|
| openai>=1.30.0 | LLM 客户端（OpenAI 兼容 API 调 SGLang） |
| fastapi>=0.111.0 | 检索服务 HTTP 框架 |
| uvicorn>=0.30.0 | ASGI 服务器 |
| pydantic>=2.7.0 | 数据校验 |
| Pillow>=10.3.0 | 图像处理 |
| numpy>=1.26.0 | 数值计算 |
| datasets>=2.19.0 | HuggingFace 数据集加载 |
| faiss-cpu>=1.7.4 | 向量检索（本地 + 检索服务） |
| sentence-transformers>=2.7.0 | 文本嵌入 |
| pyyaml>=6.0.1 | 配置解析 |
| scipy>=1.13.0 | 科学计算 |
| rank-bm25 | BM25 关键词检索（检索服务用） |
| httpx | 异步 HTTP 客户端（主 pipeline 调远程检索） |

可选依赖组：
```bash
pip install -e ".[eval]"    # 评估指标：rouge-score, bert-score
pip install -e ".[ingest]"  # 数据导入：owlready2, biopython, lxml
pip install -e ".[dev]"     # 开发：pytest, pytest-asyncio
```

### 3.3 GPU 需求

| 组件 | GPU 需求 | 说明 |
|------|---------|------|
| SGLang 模型服务 | 1× A100 80GB | Qwen3.5-35B-A3B, bfloat16, mem_fraction=0.85 |
| 检索服务 embedding | 1× GPU (任意) | S-PubMedBert-MS-MARCO, ~400MB 显存 |
| 评估 (GREEN) | 1× GPU | RadLlama2-7b, ~14GB 显存 |
| 评估 (RaTEScore) | 1× GPU | ~4GB 显存 |

如果 GPU 有限，embedding 可以用 CPU（慢但可行），评估指标可以分批跑。

### 3.4 外部数据

| 数据 | 路径 | 说明 |
|------|------|------|
| CheXpert-plus-RRG | `/mnt/yulan_pretrain_131/clinical_project/hf/data/Doorad/CheXpert-plus-RRG` | HuggingFace parquet，含胸片图像+报告文本 |
| Qwen3.5-35B-A3B | `/mnt/yulan_pretrain_131/clinical_project/hf/Qwen/Qwen3.5-35B-A3B` | LLM 模型权重 |
| S-PubMedBert-MS-MARCO | `/mnt/yulan_pretrain_131/clinical_project/hf` 或 HuggingFace 下载 | Embedding 模型 |
| GuidelineBase | 远程服务器上 | 5 类指南 JSONL |
| CaseBase | 远程服务器上 | MIMIC-CXR 等案例 JSON |

---

## 4. 配置文件详解 (config/settings.yaml)

```yaml
model:
  path: "/mnt/yulan_pretrain_131/clinical_project/hf/Qwen/Qwen3.5-35B-A3B"  # 模型路径（仅供参考，实际由 SGLang 加载）
  base_url: "http://localhost:9000/v1"  # SGLang OpenAI 兼容 API 地址
  api_key: "EMPTY"                       # SGLang 不需要真实 key
  max_tokens: 8192                       # 单次生成最大 token 数
  temperature: 1.0                       # 采样温度
  top_p: 0.95
  top_k: 20
  presence_penalty: 1.5                  # 重复惩罚
  enable_thinking: true                  # 启用 Qwen3 思考模式
  max_retries: 3                         # LLM 调用失败重试次数
  retry_base_delay_ms: 500               # 重试基础延迟

data:
  chexpert_path: "/mnt/yulan_pretrain_131/clinical_project/hf/data/Doorad/CheXpert-plus-RRG"  # 数据集路径
  default_split: "findings_section"      # 默认数据集 split
  knowledge_dir: "./data/knowledge"      # 本地知识库目录
  indexes_dir: "./data/indexes"          # FAISS 索引缓存目录
  raw_dir: "./data/raw"                  # 原始数据目录
  output_dir: "./output"                 # 输出目录

embedding:
  model_name: "pritamdeka/S-PubMedBert-MS-MARCO"  # 嵌入模型
  device: "cuda:1"                                 # 嵌入模型所用 GPU
  batch_size: 64                                   # 编码批大小

pubmed:
  api_key: "168ef570fb9e51dc79d0cee08bd779957909"  # NCBI API key
  max_results_per_query: 5                          # 每次查询最多返回条数
  enabled: true                                     # 是否启用 PubMed 检索

snomed:
  mode: "local"                          # "local" 用本地 FAISS, "api" 用 Snowstorm
  snowstorm_url: "https://tx.fhir.org/r4"
  edition: "900000000000207008"

pipeline:
  max_iterations: 3                      # QR 反馈循环最大迭代次数

agents:                                  # 各 Agent 参数覆盖
  visual_analysis:
    max_turns: 50                        # 最大工具调用轮次
    temperature: 0.7
    max_tokens: 8192
  evidence_retrieval:
    max_turns: 20
    temperature: 0.7
    max_tokens: 4096
  report_writing:
    max_turns: 20
    temperature: 0.7
    max_tokens: 8192
  quality_review:
    max_turns: 15
    temperature: 0.7
    max_tokens: 4096

hooks:                                   # 质量钩子阈值
  min_findings: 2                        # 最少 findings 数
  min_evidence_per_finding: 1            # 每个 finding 最少证据数
  min_impressions: 1                     # 最少 impression 数
  require_anatomy_coverage: true         # 是否要求解剖覆盖

limits:
  compact_threshold_tokens: 20000        # 上下文压缩阈值
  max_image_evidence_per_session: 30     # 单次会话最大图像证据数

server:                                  # Web demo 服务器配置
  host: "0.0.0.0"
  port: 8080
  cors_origins: "*"

remote_kb:                               # 远程检索服务配置
  base_url: ""                           # 检索服务地址，如 "http://your-server:8081"
  timeout: 10.0                          # 请求超时秒数
```

**重要**：`remote_kb.base_url` 为空时，literature_search 和 case_retrieval 工具会跳过远程检索，仅使用 PubMed（如果启用）。填入地址后自动启用远程检索。

---

## 5. 启动流程

### 5.0 环境准备
conda activate radclaw

### 5.1 启动 LLM 模型服务（SGLang）或者使用API
<!-- 167cuda版本太低，无法serve qwen3.5 -->
```bash
cd /mnt/yulan_pretrain_131/clinical_project/radclaw
bash serve_sglang.sh
```

脚本内容：
```bash
export GLOO_SOCKET_IFNAME=eth0
python3 -m sglang.launch_server --model-path /mnt/yulan_pretrain_131/clinical_project/hf/Qwen/Qwen3.5-35B-A3B \
    --tp 1 \
    --port 9000 \
    --context-length 65536 \
    --trust-remote-code \
    --dtype bfloat16 \
    --mem-fraction-static 0.85 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder
```

参数说明：
- `--tp 1`：单卡张量并行（1×A100 80GB 足够 35B-A3B MoE 模型）
- `--context-length 65536`：最大上下文长度
- `--mem-fraction-static 0.85`：GPU 显存占用比例
- `--reasoning-parser qwen3`：启用 Qwen3 思考模式解析
- `--tool-call-parser qwen3_coder`：启用工具调用解析

启动后等待输出 `The server is fired up and ready to roll!`，服务在 `http://localhost:9000/v1` 提供 OpenAI 兼容 API。

验证：
```bash
curl http://localhost:9000/v1/models
```

### 5.2 启动远程检索服务（可选，在远程服务器上）

```bash
cd /mnt/yulan_pretrain_131/clinical_project
python -m retrieval_service.app --config retrieval_service/config_example.yaml
```

启动过程：
1. 加载 embedding 模型（S-PubMedBert-MS-MARCO）
2. 逐类别构建/加载 FAISS + BM25 索引
3. 索引构建完成后缓存到 `cache_dir`，下次启动直接加载
4. 服务就绪后监听 `host:port`（默认 0.0.0.0:8081）

首次启动时间取决于数据量：
- Guidelines（~几千条 chunks）：1-2 分钟
- Cases（MIMIC-CXR 225K）：GPU batch_size=1024 约 15-25 分钟

后续启动直接从缓存加载，几秒内就绪。

### 5.3 运行 Pipeline

确保 SGLang 已启动（或在线 API 已配置），然后：

#### run_single.py — 跑单条样本

全流程（4-Agent pipeline），输出结果 + 终端打印报告对比。

```bash
python scripts/run_single.py --index 5 --settings config/settings.yaml

# 参数：
#   --index N          样本索引（默认 0）
#   --split NAME       数据集 split（默认 findings_section）
#   --output-dir DIR   输出目录（默认 ./output）
#   --settings PATH    配置文件（默认 config/settings.yaml）
#
# 输出文件：{output-dir}/sample_{index}_{split}.json
# 终端打印：生成报告 vs ground truth 对比 + 统计
```

#### run_trace.py — 批量跑 + 保存完整 Agent 轨迹

每个样本保存：事件时间线、每个 Agent 的详细 tool 调用记录、图像证据文件。
**适合调试和分析 Agent 行为。**

```bash
python scripts/run_trace.py --indices 0,1,2,3,4 --output-dir output/traces_v1

# 参数：
#   --indices I,J,K    逗号分隔的样本索引（默认 1,2,3）
#   --split NAME       数据集 split（默认 findings_section）
#   --output-dir DIR   输出目录（默认 ./output/traces）
#   --settings PATH    配置文件（默认 config/settings.yaml）
#
# 输出结构：
#   {output-dir}/
#   ├── batch_summary.json                # 批次汇总
#   └── sample_{index}/
#       ├── summary.json                  # 样本汇总（evaluate.py 读此文件）
#       ├── result.json                   # 完整 pipeline 结果
#       ├── event_timeline.json           # 全事件流
#       ├── trace_0_visual_analysis.json  # VA Agent 完整交互
#       ├── trace_0_evidence_retrieval.json
#       ├── trace_0_report_writing.json
#       ├── trace_0_quality_review.json
#       └── images/                       # 图像证据 PNG
```

#### run_batch.py — 批量跑（轻量输出）

只保存最终结果 JSON，不记录详细轨迹。**适合大批量跑评估数据。**

```bash
python scripts/run_batch.py --start 0 --end 62 --output-dir output/batch_v1

# 参数：
#   --start N          起始索引（默认 0）
#   --end N            结束索引，-1 表示全部（默认 -1）
#   --split NAME       数据集 split（默认 findings_section）
#   --output-dir DIR   输出目录（默认 ./output/batch）
#   --settings PATH    配置文件（默认 config/settings.yaml）
#
# 输出结构：
#   {output-dir}/
#   ├── batch_summary.json     # 批次统计
#   └── sample_{index}.json    # 每个样本的结果
#
# 跑全集：
python scripts/run_batch.py --end -1 --output-dir output/batch_full
# 或指定范围：
python scripts/run_batch.py --start 30 --end 62 --output-dir output/batch_full
```

#### run_ablation.py — 消融实验（B1/B2/B3/Full）

支持 4 种模式对比，不指定 `--indices` 默认跑全集。详见下面 §8。

```bash
python scripts/run_ablation.py --mode full --output-dir output/ablation_full

# 参数：
#   --mode MODE        direct|visual_only|retrieval_only|full（默认 direct）
#   --indices I,J,K    逗号分隔索引（不指定则跑全集）
#   --split NAME       数据集 split（默认 findings_section）
#   --output-dir DIR   输出目录（默认 output/ablation_{mode}）
#   --settings PATH    配置文件（默认 config/settings.yaml）
```

#### 脚本对比

| 脚本 | 用途 | 输出详细度 | 是否支持消融 |
|------|------|-----------|------------|
| run_single.py | 调试单条 | 低（仅结果 JSON） | 否（仅 full） |
| run_trace.py | 分析 Agent 行为 | 高（完整轨迹） | 否（仅 full） |
| run_batch.py | 大批量跑数据 | 低（仅结果 JSON） | 否（仅 full） |
| run_ablation.py | 消融对比实验 | 中（summary + result + timeline） | 是（4 种模式） |

---

## 6. Pipeline 架构详解

### 6.1 四个 Agent

```
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Visual Analysis │ ──→ │ Evidence Retrieval   │ ──→ │ Report Writing  │ ──→ │ Quality Review  │
│ (8 tools)       │     │ (7 tools)            │     │ (5 tools)       │     │ (6 tools)       │
└─────────────────┘     └─────────────────────┘     └─────────────────┘     └────────┬────────┘
                                                                                      │
                                                                              ┌───────▼───────┐
                                                                              │ review_passed? │
                                                                              └───────┬───────┘
                                                                         yes ←─┘           └─→ no
                                                                         │                      │
                                                                      [DONE]            [重跑部分 Agent]
                                                                                    (最多 max_iterations 次)
```

**反馈循环机制**：QR Agent 的 `attribute_issue` 工具会标注问题归属（visual/retrieval/generation/wording），orchestrator 据此决定从哪个 Agent 开始重跑。例如 attribution="visual" 会从 VA 开始重跑整条链。

### 6.2 工具清单

#### Visual Analysis (8 tools)
| 工具 | 功能 |
|------|------|
| select_slice | 选择 CT/MRI 切片 |
| crop_roi | 裁剪感兴趣区域 |
| zoom_enhance | 放大增强局部 |
| multi_window | 多窗宽窗位显示 |
| compose_panel | 组合多图面板 |
| overlay_annotation | 叠加标注 |
| generate_probe | 生成探测图像 |
| link_evidence | 关联图像证据到 finding |

#### Evidence Retrieval (7 tools)
| 工具 | 功能 |
|------|------|
| standardize_obs | 标准化观察描述（RadLex/SNOMED 映射） |
| concept_lookup | 术语概念查询 |
| literature_search | 文献检索（远程指南 + PubMed） |
| case_retrieval | 相似案例检索（远程） |
| evidence_scoring | 证据相关性评分 |
| conflict_detection | 证据冲突检测 |
| evidence_summary | 证据汇总 |

#### Report Writing (5 tools)
| 工具 | 功能 |
|------|------|
| build_finding | 构建单条 finding |
| build_impression | 构建 impression |
| link_support | 关联支撑证据 |
| assemble_report | 组装结构化报告 |
| render_report | 渲染自然语言报告 |

#### Quality Review (6 tools)
| 工具 | 功能 |
|------|------|
| visual_consistency | 检查视觉证据一致性 |
| evidence_consistency | 检查文献证据一致性 |
| internal_consistency | 检查报告内部一致性 |
| standard_consistency | 检查是否符合指南标准 |
| attribute_issue | 标注问题归属 |
| human_feedback | 模拟人类反馈 |

### 6.3 共享状态 (PipelineState)

所有 Agent 通过 `PipelineState` 共享数据：

```python
@dataclass
class PipelineState:
    image_evidence: list[ImageEvidence]        # VA 产出的图像证据
    retrieved_evidence: list[RetrievedEvidence] # ER 产出的检索证据
    structured_report: StructuredReport | None  # RW 产出的结构化报告
    review_issues: list[ReviewIssue]           # QR 产出的问题列表
    review_passed: bool                        # QR 是否通过
    iteration_count: int                       # 当前迭代次数
    modality: str                              # 影像模态（如 "X-Ray"）
    exam_info: dict                            # 检查信息
```

### 6.4 执行引擎 (query_loop)

`src/core/loop.py` 中的 `query_loop()` 是单 Agent 的执行引擎：
- 异步生成器，yield StreamEvent
- 支持多轮工具调用（tool-use conversation loop）
- 支持视觉输入（图像 base64）
- 并行工具执行（concurrent-safe tools）
- 上下文压缩（当 token 数超过 `compact_threshold_tokens` 时自动精简历史）
- max_turns 保护（防止无限循环）

---

## 7. 远程检索服务详解

### 7.1 架构

检索服务是一个独立的 FastAPI 应用，与主 pipeline 通过 HTTP 通信。支持两类检索：

1. **Guidelines（指南）**：医学指南文本块，按类别索引
2. **Cases（案例）**：既往放射学报告，按类别索引

每类检索使用 **Dense (FAISS) + BM25 + RRF 融合**：
- Dense：S-PubMedBert-MS-MARCO 编码 → FAISS IndexFlatIP（余弦相似度）
- BM25：BM25Okapi 关键词匹配
- RRF：Reciprocal Rank Fusion 合并两路结果，归一化到 [0,1]

### 7.2 数据格式

#### Guidelines 数据

路径结构：
```
{guidelines_base}/{category}/guideline_chunks/retrieval/chunks.jsonl
```

例如：
```
/mnt/.../GuidelineBase/chest_xray/guideline_chunks/retrieval/chunks.jsonl
/mnt/.../GuidelineBase/brain_ct/guideline_chunks/retrieval/chunks.jsonl
```

每行一个 JSON 对象：
```json
{
  "chunk_id": "fleischner_2017_ch3_p2",
  "doc_id": "fleischner_2017",
  "title": "Fleischner Society 2017 Guidelines",
  "section": "Chapter 3: Nodule Management",
  "text": "For solid nodules 6-8mm, follow-up CT at 6-12 months is recommended...",
  "retrieval_text": "solid nodule 6-8mm follow-up CT recommendation Fleischner..."
}
```

字段说明：
- `chunk_id`：块唯一标识
- `doc_id`：所属文档标识
- `title`：文档标题
- `section`：所属章节
- `text`：返回给用户的完整文本
- `retrieval_text`：用于索引的文本（可能经过优化，如加入关键词）

#### Cases 数据

路径结构：
```
{cases_base}/{dataset}/{任意子目录结构}/*.json
```

例如 MIMIC-CXR：
```
/mnt/.../CaseBase/MIMIC-CXR/train/p10/p10000032/s50414267.json
/mnt/.../CaseBase/MIMIC-CXR/train/p10/p10000032/s53911762.json
/mnt/.../CaseBase/MIMIC-CXR/train/p11/p11000001/s12345678.json
```

服务会递归扫描 `{cases_base}/{dataset}/` 下所有 `*.json` 文件。

每个 JSON 文件：
```json
{
  "case_id": "s50414267",
  "sections": {
    "findings": "The cardiac silhouette is normal in size...",
    "impression": "No acute cardiopulmonary process."
  },
  "display_report": "FINDINGS: The cardiac silhouette...\nIMPRESSION: No acute...",
  "positive_findings": ["normal cardiac silhouette", "clear lungs"],
  "negative_findings": ["no pleural effusion", "no pneumothorax"],
  "retrieval_text_dense": "normal chest radiograph clear lungs no effusion...",
  "retrieval_text_bm25": "cardiac silhouette normal size lungs clear bilateral..."
}
```

字段说明：
- `case_id`：案例唯一标识（默认取文件名 stem）
- `sections.findings`：影像所见
- `sections.impression`：诊断印象
- `display_report`：完整报告文本
- `positive_findings`：阳性发现列表
- `negative_findings`：阴性发现列表
- `retrieval_text_dense`：用于 dense 检索的文本
- `retrieval_text_bm25`：用于 BM25 检索的文本（可以不同于 dense）

### 7.3 API 接口

#### GET /health

返回服务状态和已加载的索引列表。

```bash
curl http://localhost:8081/health
```

响应：
```json
{
  "status": "ok",
  "indexes": ["guidelines/chest_xray", "guidelines/brain_ct", "cases/chest_xray"]
}
```

#### POST /search/guidelines

请求体：
```json
{
  "query": "pneumonia consolidation lobar opacity",
  "category": "chest_xray",
  "max_results": 5
}
```

- `query`：检索查询文本（必填，非空）
- `category`：类别，必须是 `chest_xray`, `chest_ct`, `brain_ct`, `brain_mri`, `abdominal_ct` 之一
- `max_results`：返回结果数（1-30，默认 5）

响应：
```json
{
  "results": [
    {
      "chunk_id": "fleischner_2017_ch3_p2",
      "doc_id": "fleischner_2017",
      "title": "Fleischner Society 2017 Guidelines",
      "section": "Chapter 3: Nodule Management",
      "text": "For solid nodules 6-8mm...",
      "score": 0.9823
    }
  ]
}
```

#### POST /search/cases

请求体：
```json
{
  "query": "consolidation pleural effusion pneumonia",
  "category": "chest_xray",
  "max_results": 5
}
```

- `query`：检索查询文本（必填，非空）
- `category`：同上
- `max_results`：返回结果数（1-50，默认 5）

响应：
```json
{
  "results": [
    {
      "case_id": "s50414267",
      "sections": {
        "findings": "The cardiac silhouette is normal...",
        "impression": "No acute cardiopulmonary process."
      },
      "positive_findings": ["normal cardiac silhouette"],
      "negative_findings": ["no pleural effusion"],
      "display_report": "FINDINGS: ...\nIMPRESSION: ...",
      "score": 0.8745
    }
  ]
}
```

#### 错误码

| HTTP 状态码 | 含义 |
|------------|------|
| 400 | 无效的 category |
| 422 | 请求体校验失败（如 query 为空） |
| 503 | Embedder 未就绪（服务启动中） |

### 7.4 检索服务配置文件

`retrieval_service/config_example.yaml`：

```yaml
guidelines_base: "/mnt/yulan_pretrain_131/clinical_project/GuidelineBase"
cases_base: "/mnt/yulan_pretrain_131/clinical_project/CaseBase"

embedding_model: "pritamdeka/S-PubMedBert-MS-MARCO"
device: "cuda:0"          # embedding 模型设备，"cpu" 也可以
batch_size: 64            # 索引构建时的编码批大小（GPU 大可以开到 1024）

cache_dir: "./retrieval_cache"   # 索引缓存目录

categories:               # 要索引的类别
  - chest_xray
  - chest_ct
  - brain_ct
  - brain_mri
  - abdominal_ct

case_datasets:            # 每个类别对应的案例数据集子目录
  chest_xray:
    - "MIMIC-CXR"
  # chest_ct:
  #   - "CT-RATE"

rrf_k: 60                # RRF 常数（越大越平滑）
dense_candidates: 50     # 每路检索的候选数（RRF 前）

host: "0.0.0.0"
port: 8081
```

**关键参数**：
- `batch_size`：影响索引构建速度。A100 80GB 可以开到 1024，普通 GPU 用 64-256
- `dense_candidates`：每路检索取 top-N 候选再做 RRF 融合，50 通常足够
- `rrf_k`：RRF 公式中的常数 k，score = Σ 1/(k + rank)，k=60 是标准值
- `case_datasets`：字典，key 是 category，value 是 `cases_base` 下的子目录列表。服务会递归扫描这些目录下所有 `.json` 文件

### 7.5 索引缓存机制

首次构建后，索引保存为 3 个文件：
```
{cache_dir}/guidelines_{category}.faiss      # FAISS 向量索引
{cache_dir}/guidelines_{category}_bm25.pkl   # BM25 模型（pickle）
{cache_dir}/guidelines_{category}_docs.pkl   # 文档列表（pickle）

{cache_dir}/cases_{category}.faiss
{cache_dir}/cases_{category}_bm25.pkl
{cache_dir}/cases_{category}_docs.pkl
```

如果数据更新了，需要删除对应的缓存文件重新构建：
```bash
rm retrieval_cache/cases_chest_xray.*
```

### 7.6 测试检索服务

```bash
python retrieval_service/test_service.py --url http://localhost:8081
```

测试内容：
- `/health` 端点
- 5 个类别的 guidelines 检索（每类 5 条查询）
- 5 个类别的 cases 检索（每类 2-3 条查询）
- 边界情况：无效 category → 400，空 query → 422

未加载的类别会标记为 SKIP（不算失败）。

### 7.7 部署到远程服务器

需要复制的文件：
```
retrieval_service/
├── __init__.py
├── app.py
├── config.py
├── config_example.yaml
├── index.py
├── search.py
├── run.py
└── test_service.py
```

不需要复制 `__pycache__/`。

远程服务器依赖：
```bash
pip install fastapi uvicorn pyyaml faiss-cpu sentence-transformers rank-bm25 numpy
# 如果用 GPU：pip install faiss-gpu 替代 faiss-cpu
```

启动：
```bash
cd /path/to/project_root   # 包含 retrieval_service/ 的目录
python -m retrieval_service.app --config retrieval_service/config_example.yaml --port 8081
```

---

## 8. 消融实验

### 8.1 四种模式

| 模式 | 代号 | Agent 组合 | 说明 |
|------|------|-----------|------|
| direct | B1 | 无 Agent | 单次 LLM 调用，直接 VLM 生成，无工具 |
| visual_only | B2 | VA + RW | 仅视觉分析+报告，无证据检索 |
| retrieval_only | B3 | ER + RW | 仅证据检索+报告，无视觉分析 |
| full | — | VA + ER + RW + QR | 完整 4-Agent pipeline |

**B3 说明**：ER Agent 直接接收原始图像（无 VA 的先验分析），提示词要求它先自行观察图像再检索证据，最后由 RW Agent 生成报告。

### 8.2 运行消融

```bash
# B1: 直接 VLM
python scripts/run_ablation.py --mode direct --indices 0,1,2,3,4 --output-dir output/ablation_direct

# B2: 仅视觉
python scripts/run_ablation.py --mode visual_only --indices 0,1,2,3,4 --output-dir output/ablation_visual

# B3: 仅检索
python scripts/run_ablation.py --mode retrieval_only --indices 0,1,2,3,4 --output-dir output/ablation_b3

# Full pipeline
python scripts/run_ablation.py --mode full --indices 0,1,2,3,4 --output-dir output/ablation_full
```

所有模式输出格式相同，evaluate.py 可以直接评估。

### 8.3 跑全集命令（62 samples，findings_section split）

数据集共 62 条样本（index 0-61）。不指定 `--indices` 时默认跑全集：

```bash
# ============================================================
# B1: Direct VLM（单次 LLM 调用，无 Agent，最快）
# ============================================================
nohup python scripts/run_ablation.py \
    --mode direct \
    --settings config/settings.yaml \
    --output-dir output/ablation_direct_full \
    > output/ablation_direct_full_run.log 2>&1 &

# ============================================================
# B2: Visual Only（VA + RW，无证据检索）
# ============================================================
nohup python scripts/run_ablation.py \
    --mode visual_only \
    --settings config/settings.yaml \
    --output-dir output/ablation_visual_only_full \
    > output/ablation_visual_only_full_run.log 2>&1 &

# ============================================================
# B3: Retrieval Only（ER + RW，无视觉分析）
# ============================================================
nohup python scripts/run_ablation.py \
    --mode retrieval_only \
    --settings config/settings.yaml \
    --output-dir output/ablation_retrieval_only_full \
    > output/ablation_retrieval_only_full_run.log 2>&1 &

# ============================================================
# Full: 完整 4-Agent Pipeline（VA + ER + RW + QR，最慢）
# ============================================================
nohup python scripts/run_ablation.py \
    --mode full \
    --settings config/settings.yaml \
    --output-dir output/ablation_full_all \
    > output/ablation_full_all_run.log 2>&1 &
```

如果只想跑部分样本，用 `--indices`：
```bash
python scripts/run_ablation.py --mode full --indices 0,1,2,3,4
```

**评估（跑完后）：**

```bash
# 逐个评估
# evaluate.py会清洗报告
python scripts/evaluate.py --predictions output/ablation_direct_full --metrics all
python scripts/evaluate.py --predictions output/ablation_visual_only_full --metrics all
python scripts/evaluate.py --predictions output/ablation_retrieval_only_full --metrics all
python scripts/evaluate.py --predictions output/ablation_full_all --metrics all
```

**查看进度：**

```bash
# 查看已完成多少样本
ls output/ablation_direct_full/sample_*/summary.json 2>/dev/null | wc -l
ls output/ablation_full_all/sample_*/summary.json 2>/dev/null | wc -l

# 查看运行日志尾部
tail -20 output/ablation_full_all_run.log
```

**预估耗时（单卡 A100 + SGLang）：**

| 模式 | 单样本 | 全集 62 条 |
|------|--------|-----------|
| B1 direct | ~10-15s | ~10-15 min |
| B2 visual_only | ~60-90s | ~1-1.5h |
| B3 retrieval_only | ~40-60s | ~40-60 min |
| Full pipeline | ~90-180s | ~2-3h |

**使用在线 API 时的注意事项：**

- 在线 API 有 rate limit，如果报 429 错误，代码会自动重试（指数退避）
- 如果 rate limit 严重，可以分批跑：
  ```bash
  # 分两批跑 full
  python scripts/run_ablation.py --mode full --indices "0,1,2,...,30" --output-dir output/ablation_full_all
  python scripts/run_ablation.py --mode full --indices "31,32,...,61" --output-dir output/ablation_full_all
  ```
  （同一个 output-dir，不会覆盖已有样本）

- 改用在线 API 只需修改 `config/settings.yaml`：
  ```yaml
  model:
    path: "qwen-plus"     # 或 qwen-max, qwen3-235b-a22b 等
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: "sk-your-key"
  ```

### 8.4 多数据集支持

`run_ablation.py` 支持 `--dataset` 参数切换数据集：

| 数据集 | `--dataset` 值 | 测试集大小 | 数据路径 |
|--------|---------------|-----------|---------|
| CheXpert-plus-RRG | `chexpert`（默认） | 62 | `/cpfs01/.../Doorad/CheXpert-plus-RRG` |
| IU-Xray-RRG | `iu_xray` | 3307 | `/cpfs01/.../X-iZhang/IU-Xray-RRG` |
| MIMIC-CXR-RRG | `mimic_cxr` | 2461 | `/cpfs01/.../Yamini-1628/MIMIC-CXR-RRG` |
| Open-I | `openi` | 770 | `/cpfs01/.../ykumards/open-i` |

数据格式说明：
- **IU-Xray-RRG**：与 CheXpert 格式完全一致（`main_image` + `findings_section`），impression 通过 image hash join
- **MIMIC-CXR-RRG**：字段更丰富（含 indication、view、impression 在同行），test split 2461 条
- **Open-I**：原始数据无 train/test 划分，自动以 seed=42 随机抽 20% 作为测试集（770 条）

Loader 文件：
- `src/data/chexpert_loader.py` — CheXpert（原有）
- `src/data/iu_xray_loader.py` — IU-Xray-RRG
- `src/data/mimic_cxr_loader.py` — MIMIC-CXR-RRG
- `src/data/openi_loader.py` — Open-I

**跑全集命令（各数据集 × 各模式）：**

```bash
# --- CheXpert (62 条，默认) ---
python scripts/run_ablation.py --mode full --output-dir output/full_chexpert
python scripts/run_ablation.py --mode direct --output-dir output/direct_chexpert

# --- IU-Xray (3307 条) ---
python scripts/run_ablation.py --mode full --dataset iu_xray --output-dir output/full_iuxray
python scripts/run_ablation.py --mode direct --dataset iu_xray --output-dir output/direct_iuxray

# --- MIMIC-CXR (2461 条) ---
python scripts/run_ablation.py --mode full --dataset mimic_cxr --output-dir output/full_mimic
python scripts/run_ablation.py --mode direct --dataset mimic_cxr --output-dir output/direct_mimic

# --- Open-I (770 条) ---
python scripts/run_ablation.py --mode full --dataset openi --output-dir output/openi/full_openi
python scripts/run_ablation.py --mode direct --dataset openi --output-dir output/direct_openi
```

评估方式不变：
```bash
python scripts/evaluate.py --predictions output/full_iuxray --metrics all
python scripts/evaluate.py --predictions output/full_mimic --metrics all
python scripts/evaluate.py --predictions output/full_openi --metrics all
```

---

## 9. 评估系统

### 9.1 六个指标

| 指标 | 类型 | 说明 |
|------|------|------|
| BLEU-4 | NLG | n-gram 精确率（sacrebleu） |
| ROUGE-L | NLG | 最长公共子序列（rouge_score） |
| METEOR | NLG | 对齐+同义词匹配（nltk） |
| CheXbert F1 | Clinical | 14 类胸片标签 F1（f1chexbert） |
| GREEN | Clinical | 放射学报告质量评分（StanfordAIMI/GREEN-RadLlama2-7b） |
| RaTEScore | Clinical | 放射学术语实体评分 |

### 9.2 运行评估

```bash
# 评估单个输出目录
python scripts/evaluate.py --predictions output/traces/

# 指定指标子集
python scripts/evaluate.py --predictions output/traces/ --metrics bleu,rouge_l,meteor
python scripts/evaluate.py --predictions output/traces/ --metrics chexbert_f1,green,ratescore

# 评估所有指标
python scripts/evaluate.py --predictions output/traces/ --metrics all
```

### 9.3 评估输入格式

evaluate.py 扫描 `--predictions` 目录下的 `summary.json` 或 `sample_*.json` 文件，提取：
- `natural_language_report`：生成的报告文本
- `ground_truth`：参考报告文本

### 9.4 评估输出

```
{output_dir}/
├── evaluation_metrics.json      # 汇总指标
└── predictions_vs_gt.json       # 逐样本对比
```

`evaluation_metrics.json` 示例：
```json
{
  "bleu4_corpus": 0.1234,
  "bleu4_avg": 0.1456,
  "rouge_l_f": 0.3456,
  "meteor_avg": 0.2345,
  "chexbert_f1_micro": 0.5678,
  "green_avg": 0.4567,
  "ratescore_avg": 0.6789,
  "num_samples": 50
}
```

---

## 10. 输出文件格式

### 10.1 run_single.py 输出

`output/sample_{index}_{split}.json`：
```json
{
  "natural_language_report": "FINDINGS: ...\nIMPRESSION: ...",
  "structured_report": { ... },
  "image_evidence_count": 5,
  "retrieved_evidence_count": 8,
  "iterations": 2,
  "review_passed": true
}
```

### 10.2 run_trace.py 输出

```
output/traces/sample_{index}/
├── summary.json              # 汇总信息
├── result.json               # 完整结果
├── event_timeline.json       # 所有事件时间线
├── agent_visual_analysis_iter0.json    # VA Agent 轨迹
├── agent_evidence_retrieval_iter0.json # ER Agent 轨迹
├── agent_report_writing_iter0.json     # RW Agent 轨迹
├── agent_quality_review_iter0.json     # QR Agent 轨迹
└── images/                   # 提取的图像证据
```

`summary.json`：
```json
{
  "sample_index": 5,
  "ground_truth": "原始报告文本...",
  "natural_language_report": "生成的报告...",
  "iterations": 2,
  "review_passed": true,
  "total_elapsed_seconds": 45.2,
  "agent_summaries": [
    {
      "agent": "visual_analysis",
      "iteration": 0,
      "tool_calls": 12,
      "elapsed_seconds": 18.5
    }
  ]
}
```

---

## 11. 数据集详解

### 11.1 支持的数据集

| 数据集 | 路径 | 测试集 | Loader |
|--------|------|--------|--------|
| CheXpert-plus-RRG | `/cpfs01/.../Doorad/CheXpert-plus-RRG` | valid: 62 条 | `CheXpertLoader` |
| IU-Xray-RRG | `/cpfs01/.../X-iZhang/IU-Xray-RRG` | test: 3307 条 | `IUXrayLoader` |
| MIMIC-CXR-RRG | `/cpfs01/.../Yamini-1628/MIMIC-CXR-RRG` | test: 2461 条 | `MIMICCXRLoader` |
| Open-I | `/cpfs01/.../ykumards/open-i` | 770 条 (20% split, seed=42) | `OpenILoader` |

所有 loader 输出统一接口：
```python
sample = loader[idx]
sample["image"]       # PIL.Image
sample["text"]        # "FINDINGS:\n...\n\nIMPRESSION:\n..." (完整 GT)
sample["findings"]    # 单独 findings 文本
sample["impression"]  # 单独 impression 文本

loader.get_exam_info(idx)  # {"study_type", "modality", "view", ...}
```

### 11.2 CheXpert-plus-RRG

HuggingFace parquet 格式，路径：`/mnt/yulan_pretrain_131/clinical_project/hf/data/Doorad/CheXpert-plus-RRG`

可用 split：
- `findings_section`：仅含 findings 段落的报告
- `train` / `valid` / `test`：标准划分

```python
from src.data.chexpert_loader import CheXpertLoader
loader = CheXpertLoader(
    data_dir="/mnt/yulan_pretrain_131/clinical_project/hf/data/Doorad/CheXpert-plus-RRG",
    split_name="findings_section",
)
```

### 11.3 IU-Xray-RRG

与 CheXpert 格式完全一致（`main_image` + `findings_section` + `default_prompt`）。
Impression 通过 `impression_section` split 的 image hash join 获取。

```python
from src.data.iu_xray_loader import IUXrayLoader
loader = IUXrayLoader(
    data_dir="/mnt/yulan_pretrain_131/clinical_project/hf/data/X-iZhang/IU-Xray-RRG",
    split_name="findings_section",
)
# len(loader) = 3307
```

### 11.4 MIMIC-CXR-RRG

字段最丰富：findings 和 impression 在同一行，还有 indication、view、study_id 等元数据。

```python
from src.data.mimic_cxr_loader import MIMICCXRLoader
loader = MIMICCXRLoader(
    data_dir="/mnt/yulan_pretrain_131/clinical_project/hf/data/Yamini-1628/MIMIC-CXR-RRG",
    split_name="findings_section",
    split="test",
)
# len(loader) = 2461
```

### 11.5 Open-I

原始数据只有 `train` split（3851 条），图像为 `img_frontal` bytes 字段。
Loader 自动过滤无效样本（无 frontal 图或 findings 过短），以 seed=42 随机抽 20% 作为测试集。

```python
from src.data.openi_loader import OpenILoader
loader = OpenILoader(
    data_dir="/mnt/yulan_pretrain_131/clinical_project/hf/data/ykumards/open-i",
    split="test",
)
# len(loader) = 770
```

`exam_info["modality"]` 会被 orchestrator 映射到检索类别（"X-Ray" → "chest_xray"）。

---

## 12. 知识库层详解

### 12.1 三层架构

```
Terminology（本地）    Literature（远程 + PubMed）    Cases（远程）
     │                          │                        │
  RadLex FAISS              远程检索服务               远程检索服务
  SNOMED FAISS              /search/guidelines         /search/cases
     │                          │                        │
  standardize_obs           literature_search          case_retrieval
  concept_lookup
```

### 12.2 Terminology 层（本地）

**RadLex**：放射学术语本体，46,898 条术语

索引构建：
```bash
python scripts/ingest/build_indexes.py --source radlex
```

索引文件位置：`data/indexes/radlex.*`

**SNOMED**：医学术语系统

两种模式（`config/settings.yaml` 中 `snomed.mode`）：
- `local`：使用本地 FAISS 索引（需先构建）
- `api`：调用 Snowstorm FHIR API（需网络）

### 12.3 Literature 层（远程 + PubMed）

`src/tools/evidence/literature_search.py` 中的 `LiteratureSearchTool`：

1. **远程指南检索**：POST `{remote_base_url}/search/guidelines`
2. **PubMed 检索**：通过 NCBI E-utilities API

两路并行执行（`asyncio.gather`），结果合并返回。

如果 `remote_base_url` 为空，跳过远程检索，仅用 PubMed。
如果 `pubmed.enabled: false`，跳过 PubMed。

### 12.4 Cases 层（远程）

`src/tools/evidence/case_retrieval.py` 中的 `CaseRetrievalTool`：

将 ER Agent 收集到的 `observations` 列表拼接成查询字符串，POST 到 `{remote_base_url}/search/cases`。

如果 `remote_base_url` 为空，返回空结果（不报错）。

---

## 13. 常见问题排查

### Q: SGLang 启动后 pipeline 报连接错误

检查 `config/settings.yaml` 中 `model.base_url` 是否与 SGLang 端口一致：
```yaml
model:
  base_url: "http://localhost:9000/v1"
```

验证 SGLang 是否就绪：
```bash
curl http://localhost:9000/v1/models
```

### Q: 检索服务启动报 `No module named retrieval_service`

必须从项目根目录执行：
```bash
# 正确
cd /mnt/yulan_pretrain_131/clinical_project/radclaw
python -m retrieval_service.app --config retrieval_service/config_example.yaml

# 错误（不要进入 retrieval_service/ 内部执行）
cd retrieval_service && python app.py
```

### Q: 检索服务启动后没有任何输出

正常现象。启动日志输出到 stderr，用重定向查看：
```bash
python -m retrieval_service.app --config ... 2>&1 | head -50
# 或后台运行
nohup python -m retrieval_service.app --config retrieval_service/config_example.yaml \
    > retrieval.log 2>&1 &
tail -f retrieval.log
```

### Q: 索引构建很慢

增大 `batch_size`（GPU 显存允许的情况下）：
```yaml
batch_size: 1024   # A100 80GB 可用
```

MIMIC-CXR 225K 案例预估时间：
- batch_size=64（CPU）：2-5 小时
- batch_size=64（GPU）：30-60 分钟
- batch_size=1024（A100）：15-25 分钟

构建完成后缓存到 `cache_dir`，后续重启直接加载（几秒）。

### Q: embedding 模型下载失败（SSL 错误）

检索服务内置了 SSL 绕过和 HF mirror 自动切换（`app.py` 顶部）。也可手动指定：
```bash
export HF_ENDPOINT=https://hf-mirror.com
python -m retrieval_service.app --config ...
```

或提前下载到本地，配置中指定本地路径：
```yaml
embedding_model: "/path/to/local/S-PubMedBert-MS-MARCO"
```

### Q: 评估指标模型下载

GREEN 和 RaTEScore 需要额外模型：
```bash
python scripts/download_metric_models.py
```

### Q: 数据更新后索引未刷新

删除对应缓存文件，重启服务即可重建：
```bash
rm retrieval_cache/cases_chest_xray.*
rm retrieval_cache/guidelines_chest_xray.*
```

---

## 14. 完整启动流程（快速参考）

```bash
# 1. 启动 LLM 服务（后台）
cd /mnt/yulan_pretrain_131/clinical_project/radclaw
nohup bash serve_sglang.sh > sglang.log 2>&1 &
# 等待约 2-3 分钟，直到 sglang.log 出现 "server is fired up"

# 2. 启动检索服务（远程服务器，后台）
nohup python -m retrieval_service.app \
    --config retrieval_service/config_example.yaml \
    > retrieval.log 2>&1 &
# 首次需等待索引构建完成

# 3. 配置主 pipeline 连接检索服务
# 编辑 config/settings.yaml:
#   remote_kb:
#     base_url: "http://<retrieval-server-ip>:8081"

# 4. 验证
curl http://localhost:9000/v1/models
curl http://<retrieval-server>:8081/health

# 5. 跑单条样本测试
python scripts/run_single.py --index 0 --settings config/settings.yaml

# 6. 批量跑 + 评估
python scripts/run_trace.py --indices 0,1,2,3,4 --output-dir output/traces
python scripts/evaluate.py --predictions output/traces/ --metrics all
```

---

## 15. 代码关键路径速查

| 需要修改/查看 | 文件 |
|-------------|------|
| LLM 参数（温度、token 数等） | `config/settings.yaml` → `model` 节 |
| 远程检索服务地址 | `config/settings.yaml` → `remote_kb.base_url` |
| Agent 最大轮次 | `config/settings.yaml` → `agents.*.max_turns` |
| Pipeline 最大迭代 | `config/settings.yaml` → `pipeline.max_iterations` |
| 质量钩子阈值 | `config/settings.yaml` → `hooks` 节 |
| 检索服务数据路径 | `retrieval_service/config_example.yaml` |
| 检索服务索引缓存 | `retrieval_service/config_example.yaml` → `cache_dir` |
| 添加新工具 | `src/tools/{visual,evidence,report,review}/` 下新建，在 `__init__.py` 注册 |
| 修改 Agent 提示词 | `src/agents/` 下各 Agent builder |
| 修改 RRF 参数 | `retrieval_service/config_example.yaml` → `rrf_k`, `dense_candidates` |
| 添加新检索类别 | `retrieval_service/config_example.yaml` → `categories` + `case_datasets`；`retrieval_service/app.py` → `VALID_CATEGORIES` |
| 添加新消融基线 | `src/core/baselines.py` 新增函数；`scripts/run_ablation.py` → `PIPELINE_FN` 注册 |
| 运行 B3 消融 | `python scripts/run_ablation.py --mode retrieval_only --indices ...` |

