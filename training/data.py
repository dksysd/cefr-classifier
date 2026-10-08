"""
데이터 로드 및 분할 유틸리티.

torch / datasets 의존성 없이 동작하므로 테스트에서 가볍게 import 할 수 있습니다.
"""

import json
import random
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARALLEL_FILE = REPO_ROOT / "data" / "parallel_corpus" / "train.jsonl"
DEFAULT_INSTRUCTION_FILE = REPO_ROOT / "data" / "instructions" / "train.jsonl"

label2id = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}
id2label = {v: k for k, v in label2id.items()}


def read_records(path):
    """
    JSONL 파일에서 레코드를 읽습니다.

    두 가지 형식을 모두 지원합니다:
    - 표준 JSONL: 한 줄에 객체 하나 (generator_pipeline.py 출력 형식)
    - 레거시: 한 줄에 객체 배열 (기존 배포 데이터 형식)
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, list):
                records.extend(item for item in obj if item)
            elif obj:
                records.append(obj)
    return records


def load_parallel_corpus(parallel_file):
    """Parallel Corpus 데이터 로드. 같은 원문의 레벨별 버전은 같은 group을 가집니다."""
    data_samples = []
    for item in read_records(parallel_file):
        group = f"parallel::{item.get('source_text') or item.get('id')}"
        for level, version_data in item.get("versions", {}).items():
            if level in label2id and isinstance(version_data, dict) and version_data.get("text"):
                data_samples.append({
                    "text": version_data["text"],
                    "label_str": level,
                    "source": "parallel",
                    "group": group,
                })
    print(f"Parallel Corpus: {len(data_samples)}개 샘플")
    return data_samples


def load_instruction_data(instruction_file):
    """Instruction 데이터 로드. 같은 질문에 대한 레벨별 답변은 같은 group을 가집니다."""
    data_samples = []
    for item in read_records(instruction_file):
        conversation = item.get("conversation", [])
        query = next((m.get("content") for m in conversation if m.get("role") == "user"), None)
        group = f"instruction::{query or item.get('id')}"
        for msg in conversation:
            if msg.get("role") == "assistant" and msg.get("content"):
                level_used = msg.get("level_used", item.get("user_level"))
                if level_used not in label2id:
                    continue
                data_samples.append({
                    "text": msg["content"],
                    "label_str": level_used,
                    "source": "instruction",
                    "group": group,
                })
    print(f"Instruction: {len(data_samples)}개 샘플")
    return data_samples


def load_samples(parallel_file, instruction_file, mode="mixed", instruction_ratio=0.3, seed=42):
    """
    mode:
    - "parallel_only": Parallel Corpus만
    - "mixed": 두 데이터를 섞어서 학습
    - "instruction_only": Instruction만

    instruction_ratio: mixed 모드에서 사용할 instruction 질문(group) 비율 (0.0 ~ 1.0)
    """
    if mode not in ("parallel_only", "mixed", "instruction_only"):
        raise ValueError(f"알 수 없는 mode: {mode}")

    data_samples = []
    if mode in ("parallel_only", "mixed"):
        data_samples.extend(load_parallel_corpus(parallel_file))

    if mode in ("instruction_only", "mixed"):
        instruction_samples = load_instruction_data(instruction_file)
        if mode == "mixed" and instruction_ratio < 1.0:
            # 질문 단위로 샘플링하여 레벨별 답변 묶음이 유지되도록 함
            groups = sorted({s["group"] for s in instruction_samples})
            rng = random.Random(seed)
            kept = set(rng.sample(groups, int(len(groups) * instruction_ratio)))
            instruction_samples = [s for s in instruction_samples if s["group"] in kept]
            print(f"Instruction 질문 {instruction_ratio * 100:.0f}% 사용: {len(instruction_samples)}개 샘플")
        data_samples.extend(instruction_samples)

    if not data_samples:
        raise ValueError("데이터가 없습니다!")
    return data_samples


def group_train_test_split(labels, groups, test_size=0.1, seed=42):
    """
    같은 group(원문/질문)의 샘플이 train과 test에 나뉘지 않도록 분할합니다.
    레벨 분포는 StratifiedGroupKFold로 최대한 유지합니다.
    """
    n_splits = max(2, round(1 / test_size))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(labels)), labels, groups))
    return train_idx.tolist(), test_idx.tolist()
