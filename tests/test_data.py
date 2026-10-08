import json
from collections import Counter

from training.data import (
    group_train_test_split, label2id, load_instruction_data, load_parallel_corpus, load_samples, read_records
)

LEVELS = list(label2id)


def parallel_item(i):
    return {
        "id": f"parallel_{i}",
        "topic": "science",
        "source_text": f"source {i}",
        "versions": {lvl: {"text": f"text {i} {lvl}", "level": lvl} for lvl in LEVELS},
    }


def instruction_items(i):
    return [{
        "id": f"instruction_{i}_{lvl}",
        "user_level": lvl,
        "conversation": [
            {"role": "user", "content": f"question {i}?"},
            {"role": "assistant", "content": f"answer {i} {lvl}", "level_used": lvl},
        ],
        "domain": "science",
    } for lvl in LEVELS]


def write_jsonl(path, items):
    path.write_text("".join(json.dumps(x) + "\n" for x in items), encoding="utf-8")


def write_legacy(path, items):
    path.write_text(json.dumps(items) + "\n", encoding="utf-8")


def test_read_records_supports_both_formats(tmp_path):
    items = [parallel_item(i) for i in range(3)]
    write_jsonl(tmp_path / "a.jsonl", items)
    write_legacy(tmp_path / "b.jsonl", items)
    assert read_records(tmp_path / "a.jsonl") == items
    assert read_records(tmp_path / "b.jsonl") == items


def test_loaders_assign_groups(tmp_path):
    write_jsonl(tmp_path / "p.jsonl", [parallel_item(i) for i in range(2)])
    write_jsonl(tmp_path / "i.jsonl", [x for i in range(2) for x in instruction_items(i)])

    parallel = load_parallel_corpus(tmp_path / "p.jsonl")
    instruction = load_instruction_data(tmp_path / "i.jsonl")

    assert len(parallel) == 12 and len(instruction) == 12
    assert Counter(s["group"] for s in parallel) == {"parallel::source 0": 6, "parallel::source 1": 6}
    assert Counter(s["group"] for s in instruction) == {"instruction::question 0?": 6, "instruction::question 1?": 6}
    assert Counter(s["label_str"] for s in instruction) == {lvl: 2 for lvl in LEVELS}


def test_mixed_mode_samples_whole_instruction_groups(tmp_path):
    write_jsonl(tmp_path / "p.jsonl", [parallel_item(i) for i in range(5)])
    write_jsonl(tmp_path / "i.jsonl", [x for i in range(10) for x in instruction_items(i)])

    samples = load_samples(tmp_path / "p.jsonl", tmp_path / "i.jsonl", mode="mixed", instruction_ratio=0.3)
    instruction_groups = Counter(s["group"] for s in samples if s["source"] == "instruction")
    assert len(instruction_groups) == 3
    assert set(instruction_groups.values()) == {6}


def test_group_split_has_no_leakage():
    labels, groups = [], []
    for g in range(100):
        for lvl in range(6):
            labels.append(lvl)
            groups.append(f"g{g}")

    train_idx, test_idx = group_train_test_split(labels, groups, test_size=0.1, seed=42)

    assert len(train_idx) + len(test_idx) == len(labels)
    assert not {groups[i] for i in train_idx} & {groups[i] for i in test_idx}
    assert 0.05 < len(test_idx) / len(labels) < 0.15
    assert set(Counter(labels[i] for i in test_idx).values()) == {len(test_idx) // 6}
