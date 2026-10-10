#!/usr/bin/env bash
# setup_new_machine.sh — 新机器一键下载所有数据和模型
#
# 用法:
#   bash scripts/setup_new_machine.sh                    # 交互式，逐步确认
#   bash scripts/setup_new_machine.sh --all              # 全部下载（跳过确认）
#   bash scripts/setup_new_machine.sh --step radlex      # 只跑某一步
#   bash scripts/setup_new_machine.sh --step snomed
#   bash scripts/setup_new_machine.sh --step indexes
#   bash scripts/setup_new_machine.sh --step model
#   bash scripts/setup_new_machine.sh --step dataset
#   bash scripts/setup_new_machine.sh --step embedding
#   bash scripts/setup_new_machine.sh --step eval_models
#
# 前提:
#   1. 已粘贴代码（src/, scripts/, config/ 等存在）
#   2. 已 pip install -e ".[ingest]"
#   3. 当前目录为 radclaw 项目根目录

set -euo pipefail

# ============================================================
# 配置区（根据新机器实际路径修改）
# ============================================================

# 模型存放路径
MODEL_DIR="${MODEL_DIR:-/path/to/models/Qwen/Qwen3.5-35B-A3B}"

# CheXpert 数据集存放路径
DATASET_DIR="${DATASET_DIR:-/path/to/data/Doorad/CheXpert-plus-RRG}"

# BioPortal API key（下载 RadLex 需要，免费注册: https://bioportal.bioontology.org/accounts/new）
BIOPORTAL_API_KEY="${BIOPORTAL_API_KEY:-}"

# HuggingFace 镜像（国内设为 https://hf-mirror.com）
HF_ENDPOINT="${HF_ENDPOINT:-}"

# 构建索引用的设备
INDEX_DEVICE="${INDEX_DEVICE:-cpu}"
INDEX_BATCH_SIZE="${INDEX_BATCH_SIZE:-64}"

# ============================================================
# 内部变量
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ALL=false
RUN_STEP=""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================
# 参数解析
# ============================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)       RUN_ALL=true; shift ;;
        --step)      RUN_STEP="$2"; shift 2 ;;
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
        --bioportal-key) BIOPORTAL_API_KEY="$2"; shift 2 ;;
        --hf-mirror) HF_ENDPOINT="$2"; shift 2 ;;
        --device)    INDEX_DEVICE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash scripts/setup_new_machine.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --all                下载全部（跳过确认）"
            echo "  --step STEP          只执行指定步骤"
            echo "  --model-dir PATH     模型存放路径 (default: $MODEL_DIR)"
            echo "  --dataset-dir PATH   CheXpert 数据集路径 (default: $DATASET_DIR)"
            echo "  --bioportal-key KEY  BioPortal API key"
            echo "  --hf-mirror URL      HuggingFace 镜像 (e.g. https://hf-mirror.com)"
            echo "  --device DEVICE      索引构建设备 (default: cpu)"
            echo ""
            echo "Steps: radlex, snomed, indexes, model, dataset, embedding, eval_models"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================
# 工具函数
# ============================================================

log_info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

confirm() {
    if $RUN_ALL; then return 0; fi
    echo ""
    echo -e "${YELLOW}>>> $1${NC}"
    read -rp "执行? [Y/n] " ans
    case "$ans" in
        [nN]*) return 1 ;;
        *) return 0 ;;
    esac
}

check_project_dir() {
    cd "$PROJECT_DIR"
    if [[ ! -f pyproject.toml ]]; then
        log_error "当前目录不是 radclaw 项目根目录: $PROJECT_DIR"
        log_error "请先粘贴代码，然后在项目根目录下运行此脚本"
        exit 1
    fi
    log_info "项目目录: $PROJECT_DIR"
}

setup_hf_mirror() {
    if [[ -n "$HF_ENDPOINT" ]]; then
        export HF_ENDPOINT
        log_info "HuggingFace 镜像: $HF_ENDPOINT"
    fi
}

ensure_dirs() {
    mkdir -p data/raw data/indexes data/knowledge/terminology
}

should_run() {
    local step="$1"
    if [[ -n "$RUN_STEP" ]]; then
        [[ "$RUN_STEP" == "$step" ]]
    else
        return 0
    fi
}

# ============================================================
# 步骤 1: 下载 RadLex OWL → 解析
# ============================================================

step_radlex() {
    if [[ -f data/raw/radlex_parsed.json ]]; then
        local count
        count=$(python3 -c "import json; print(len(json.load(open('data/raw/radlex_parsed.json'))))")
        log_ok "RadLex 已存在: $count entries"
        return 0
    fi

    if ! confirm "下载并解析 RadLex OWL (需要 BioPortal API key, 下载 ~62MB)"; then
        log_warn "跳过 RadLex"
        return 0
    fi

    if [[ -z "$BIOPORTAL_API_KEY" ]]; then
        log_error "需要 BIOPORTAL_API_KEY 环境变量"
        log_error "免费注册: https://bioportal.bioontology.org/accounts/new"
        read -rp "输入 API key (或回车跳过): " BIOPORTAL_API_KEY
        if [[ -z "$BIOPORTAL_API_KEY" ]]; then
            log_warn "跳过 RadLex"
            return 0
        fi
    fi

    log_info "下载 RadLex OWL ..."
    BIOPORTAL_API_KEY="$BIOPORTAL_API_KEY" python3 scripts/ingest/download_radlex.py --output-dir data/raw

    if [[ ! -f data/raw/radlex.owl ]]; then
        log_error "RadLex OWL 下载失败"
        return 1
    fi

    log_info "解析 RadLex OWL → radlex_parsed.json ..."
    python3 scripts/ingest/parse_radlex_owl.py \
        --owl-path data/raw/radlex.owl \
        --output data/raw/radlex_parsed.json

    cp data/raw/radlex_parsed.json data/knowledge/terminology/radlex.json
    log_ok "RadLex 完成"
}

# ============================================================
# 步骤 2: 下载 SNOMED（通过 FHIR API）
# ============================================================

step_snomed() {
    if [[ -f data/raw/snomed_parsed.json ]]; then
        local count
        count=$(python3 -c "import json; print(len(json.load(open('data/raw/snomed_parsed.json'))))")
        log_ok "SNOMED 已存在: $count entries"
        return 0
    fi

    if ! confirm "通过 FHIR API 获取 SNOMED 数据 (需要外网, 约 30-60 分钟)"; then
        log_warn "跳过 SNOMED"
        return 0
    fi

    log_info "获取 SNOMED 概念 ..."
    python3 scripts/ingest/download_snomed.py \
        --output data/raw/snomed_parsed.json \
        --max-per-query 100

    # 生成 terminology/snomed.json
    log_info "生成 terminology/snomed.json ..."
    python3 -c "
import json
with open('data/raw/snomed_parsed.json') as f:
    entries = json.load(f)
out = {
    'system': 'SNOMED CT',
    'title': 'SNOMED Clinical Terms — Radiology Subset (via FHIR)',
    'source': 'SNOMED International / NLM',
    'version': 'fhir-api',
    'total_entries': len(entries),
    'entries': entries,
}
with open('data/knowledge/terminology/snomed.json', 'w') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f'Written {len(entries)} entries')
"
    log_ok "SNOMED 完成"
}

# ============================================================
# 步骤 3: 重建 FAISS + BM25 索引（仅 radlex + snomed）
# ============================================================

step_indexes() {
    local need_build=false

    if [[ ! -f data/indexes/radlex.index ]]; then need_build=true; fi
    if [[ ! -f data/indexes/snomed.index ]]; then need_build=true; fi

    if ! $need_build; then
        log_ok "FAISS 索引已存在"
        ls -lh data/indexes/*.index 2>/dev/null
        return 0
    fi

    if ! confirm "重建 FAISS + BM25 索引 (radlex + snomed, 设备: $INDEX_DEVICE)"; then
        log_warn "跳过索引重建"
        return 0
    fi

    if [[ ! -f data/raw/radlex_parsed.json ]]; then
        log_error "缺少 data/raw/radlex_parsed.json, 请先运行 radlex 步骤"
        return 1
    fi

    if [[ -f data/raw/radlex_parsed.json ]]; then
        log_info "构建 RadLex 索引 ..."
        python3 scripts/ingest/build_indexes.py \
            --source radlex --device "$INDEX_DEVICE" --batch-size "$INDEX_BATCH_SIZE"
    fi

    if [[ -f data/raw/snomed_parsed.json ]]; then
        log_info "构建 SNOMED 索引 ..."
        python3 scripts/ingest/build_indexes.py \
            --source snomed --device "$INDEX_DEVICE" --batch-size "$INDEX_BATCH_SIZE"
    else
        log_warn "snomed_parsed.json 不存在, 跳过 SNOMED 索引"
    fi

    log_ok "索引重建完成"
    ls -lh data/indexes/ 2>/dev/null
}

# ============================================================
# 步骤 4: 下载 Qwen3.5-35B-A3B 模型 (~68GB)
# ============================================================

step_model() {
    if [[ -d "$MODEL_DIR" ]] && [[ -f "$MODEL_DIR/config.json" ]]; then
        log_ok "模型已存在: $MODEL_DIR"
        return 0
    fi

    if ! confirm "下载 Qwen3.5-35B-A3B (~68GB) → $MODEL_DIR"; then
        log_warn "跳过模型下载"
        return 0
    fi

    if [[ "$MODEL_DIR" == "/path/to/models/Qwen/Qwen3.5-35B-A3B" ]]; then
        log_error "请先设置 MODEL_DIR 环境变量或使用 --model-dir 参数"
        read -rp "输入模型存放路径: " MODEL_DIR
        if [[ -z "$MODEL_DIR" ]]; then
            log_warn "跳过模型下载"
            return 0
        fi
    fi

    mkdir -p "$(dirname "$MODEL_DIR")"

    log_info "下载 Qwen/Qwen3.5-35B-A3B → $MODEL_DIR ..."
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download Qwen/Qwen3.5-35B-A3B --local-dir "$MODEL_DIR"
    else
        log_info "huggingface-cli 未安装, 尝试 python 方式 ..."
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-35B-A3B', local_dir='$MODEL_DIR')
"
    fi

    log_ok "模型下载完成: $MODEL_DIR"
}

# ============================================================
# 步骤 5: 下载 CheXpert 数据集 (~613MB)
# ============================================================

step_dataset() {
    if [[ -d "$DATASET_DIR" ]]; then
        local parquet_count
        parquet_count=$(find "$DATASET_DIR" -name "*.parquet" 2>/dev/null | wc -l)
        if [[ "$parquet_count" -gt 0 ]]; then
            log_ok "CheXpert 数据集已存在: $DATASET_DIR ($parquet_count parquet files)"
            return 0
        fi
    fi

    if ! confirm "下载 CheXpert-plus-RRG (~613MB) → $DATASET_DIR"; then
        log_warn "跳过数据集下载"
        return 0
    fi

    if [[ "$DATASET_DIR" == "/path/to/data/Doorad/CheXpert-plus-RRG" ]]; then
        log_error "请先设置 DATASET_DIR 环境变量或使用 --dataset-dir 参数"
        read -rp "输入数据集存放路径: " DATASET_DIR
        if [[ -z "$DATASET_DIR" ]]; then
            log_warn "跳过数据集下载"
            return 0
        fi
    fi

    mkdir -p "$(dirname "$DATASET_DIR")"

    log_info "下载 Doorad/CheXpert-plus-RRG → $DATASET_DIR ..."
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download Doorad/CheXpert-plus-RRG \
            --repo-type dataset \
            --local-dir "$DATASET_DIR"
    else
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Doorad/CheXpert-plus-RRG', repo_type='dataset', local_dir='$DATASET_DIR')
"
    fi

    log_ok "CheXpert 下载完成: $DATASET_DIR"
}

# ============================================================
# 步骤 6: 预下载 Embedding 模型
# ============================================================

step_embedding() {
    log_info "检查 embedding 模型 pritamdeka/S-PubMedBert-MS-MARCO ..."

    python3 -c "
import os, sys
try:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO', device='cpu')
    print(f'Embedding model loaded: dim={m.get_sentence_embedding_dimension()}')
except Exception as e:
    print(f'Failed: {e}', file=sys.stderr)
    sys.exit(1)
"

    if [[ $? -eq 0 ]]; then
        log_ok "Embedding 模型已就绪"
    else
        log_error "Embedding 模型下载失败"
        return 1
    fi
}

# ============================================================
# 步骤 7: 下载评估模型（可选）
# ============================================================

step_eval_models() {
    if ! confirm "下载评估模型 (CheXbert ~880MB + RaTEScore ~1.2GB, GREEN ~14GB 可选)"; then
        log_warn "跳过评估模型"
        return 0
    fi

    log_info "下载 CheXbert + RaTEScore 模型 ..."
    python3 scripts/download_metric_models.py

    echo ""
    read -rp "是否也下载 GREEN (~14GB)? [y/N] " ans
    case "$ans" in
        [yY]*)
            log_info "下载 GREEN ..."
            python3 scripts/download_metric_models.py --green
            ;;
        *)
            log_info "跳过 GREEN"
            ;;
    esac

    log_ok "评估模型下载完成"
}

# ============================================================
# 步骤 8: 自动更新配置文件
# ============================================================

update_config() {
    if ! confirm "自动更新 config/settings.yaml 和 serve_sglang.sh 中的路径"; then
        log_warn "跳过配置更新，请手动修改"
        return 0
    fi

    if [[ "$MODEL_DIR" != "/path/to/models/Qwen/Qwen3.5-35B-A3B" ]]; then
        log_info "更新 config/settings.yaml: model.path → $MODEL_DIR"
        sed -i "s|path:.*Qwen3.5-35B-A3B.*|path: \"$MODEL_DIR\"|" config/settings.yaml

        log_info "更新 serve_sglang.sh: --model-path → $MODEL_DIR"
        sed -i "s|--model-path [^ ]*|--model-path $MODEL_DIR|" serve_sglang.sh
    else
        log_warn "MODEL_DIR 未设置，跳过 model.path 更新"
    fi

    if [[ "$DATASET_DIR" != "/path/to/data/Doorad/CheXpert-plus-RRG" ]]; then
        log_info "更新 config/settings.yaml: data.chexpert_path → $DATASET_DIR"
        sed -i "s|chexpert_path:.*|chexpert_path: \"$DATASET_DIR\"|" config/settings.yaml
    else
        log_warn "DATASET_DIR 未设置，跳过 chexpert_path 更新"
    fi

    # remote_kb.base_url
    log_info "更新 config/settings.yaml: remote_kb.base_url → http://localhost:8081"
    sed -i 's|base_url: ""|base_url: "http://localhost:8081"|' config/settings.yaml

    log_ok "配置更新完成"
}

# ============================================================
# 验证
# ============================================================

verify() {
    echo ""
    echo "============================================================"
    echo "  环境验证"
    echo "============================================================"

    echo ""
    log_info "1. 核心依赖 ..."
    python3 -c "
import openai, fastapi, uvicorn, pydantic, PIL, numpy, datasets
import pyarrow, yaml, scipy, faiss, sentence_transformers
print('  Core dependencies OK')
" 2>/dev/null && log_ok "核心依赖 OK" || log_error "核心依赖缺失"

    echo ""
    log_info "2. 知识库数据 ..."
    python3 -c "
import json, os
for name, path in [('RadLex', 'data/raw/radlex_parsed.json'), ('SNOMED', 'data/raw/snomed_parsed.json')]:
    if os.path.exists(path):
        with open(path) as f:
            print(f'  {name}: {len(json.load(f))} entries')
    else:
        print(f'  {name}: NOT FOUND')
" 2>/dev/null

    echo ""
    log_info "3. FAISS 索引 ..."
    if [[ -f data/indexes/radlex.index ]]; then
        log_ok "RadLex 索引存在"
    else
        log_warn "RadLex 索引缺失"
    fi
    if [[ -f data/indexes/snomed.index ]]; then
        log_ok "SNOMED 索引存在"
    else
        log_warn "SNOMED 索引缺失"
    fi

    echo ""
    log_info "4. 模型 ..."
    if [[ -f "$MODEL_DIR/config.json" ]]; then
        log_ok "Qwen3.5-35B-A3B: $MODEL_DIR"
    else
        log_warn "模型不存在: $MODEL_DIR"
    fi

    echo ""
    log_info "5. 数据集 ..."
    local parquet_count
    parquet_count=$(find "$DATASET_DIR" -name "*.parquet" 2>/dev/null | wc -l)
    if [[ "$parquet_count" -gt 0 ]]; then
        log_ok "CheXpert: $DATASET_DIR ($parquet_count parquet files)"
    else
        log_warn "CheXpert 不存在: $DATASET_DIR"
    fi

    echo ""
    log_info "6. GPU ..."
    python3 -c "
import torch
print(f'  CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem/1e9:.1f}GB)')
" 2>/dev/null

    echo ""
    log_info "7. abbreviations.json ..."
    if [[ -f data/knowledge/abbreviations.json ]]; then
        log_ok "abbreviations.json 存在"
    else
        log_error "abbreviations.json 缺失！请检查代码粘贴步骤"
    fi

    echo ""
    echo "============================================================"
    echo "  验证完成"
    echo "============================================================"
}

# ============================================================
# 主流程
# ============================================================

main() {
    echo "============================================================"
    echo "  RadClaw 新机器环境搭建"
    echo "============================================================"
    echo ""
    echo "  MODEL_DIR:    $MODEL_DIR"
    echo "  DATASET_DIR:  $DATASET_DIR"
    echo "  INDEX_DEVICE: $INDEX_DEVICE"
    echo "  HF_ENDPOINT:  ${HF_ENDPOINT:-<default>}"
    echo ""

    check_project_dir
    setup_hf_mirror
    ensure_dirs

    if should_run "radlex";      then step_radlex;      fi
    if should_run "snomed";       then step_snomed;       fi
    if should_run "indexes";      then step_indexes;      fi
    if should_run "model";        then step_model;        fi
    if should_run "dataset";      then step_dataset;      fi
    if should_run "embedding";    then step_embedding;    fi
    if should_run "eval_models";  then step_eval_models;  fi

    if [[ -z "$RUN_STEP" ]]; then
        update_config
        verify
    fi

    echo ""
    log_ok "全部完成！"
    echo ""
    echo "下一步:"
    echo "  1. 部署 retrieval_service:"
    echo "     cp retrieval_service/config_example.yaml retrieval_service/config.yaml"
    echo "     vi retrieval_service/config.yaml  # 修改 guidelines_base 和 cases_base"
    echo "     nohup python -m retrieval_service.app --config retrieval_service/config.yaml > nohup_retrieval.out 2>&1 &"
    echo ""
    echo "  2. 启动 SGLang:"
    echo "     nohup bash serve_sglang.sh > nohup_sglang.out 2>&1 &"
    echo "     tail -f nohup_sglang.out  # 等待 'server is ready'"
    echo ""
    echo "  3. 端到端测试:"
    echo "     python scripts/run_single.py --index 0 --settings config/settings.yaml"
}

main
