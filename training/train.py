"""
CEFR Router V3 - Grid Search 실험

모든 하이퍼파라미터 조합을 자동으로 실험:
- LoRA 파라미터 (rank, alpha, dropout, target_modules)
- Learning rate
- LR Scheduler
- Instruction ratio
"""

import os
import json
import numpy as np
from itertools import product
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
import torch
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import loss functions
try:
    from loss import OrdinalLabelSmoothingLoss, FocalLoss
except ImportError:
    # Fallback for when running from root directory
    from training.loss import OrdinalLabelSmoothingLoss, FocalLoss

os.environ["TOKENIZERS_PARALLELISM"] = "false"

label2id = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}
id2label = {v: k for k, v in label2id.items()}


def load_parallel_corpus(parallel_file):
    """Parallel Corpus 데이터 로드"""
    data_samples = []

    print(f"'{parallel_file}' 파일 처리 중...")
    try:
        with open(parallel_file, 'r', encoding='utf-8') as f:
            for line in f:
                for item in json.loads(line.strip()):
                    if not item:
                        continue
                    versions = item.get("versions", {})
                    for level, version_data in versions.items():
                        if level in label2id and isinstance(version_data, dict) and "text" in version_data:
                            data_samples.append({
                                "text": version_data["text"],
                                "label_str": level,
                                "source": "parallel"
                            })
    except Exception as e:
        print(f"오류: {e}")

    print(f"Parallel Corpus: {len(data_samples)}개 샘플")
    return data_samples


def load_instruction_data(instruction_file):
    """Instruction 데이터 로드"""
    data_samples = []

    print(f"'{instruction_file}' 파일 처리 중...")
    try:
        with open(instruction_file, 'r', encoding='utf-8') as f:
            for line in f:
                for item in json.loads(line.strip()):
                    if not item:
                        continue

                    user_level = item.get("user_level")
                    conversation = item.get("conversation", [])

                    for msg in conversation:
                        if msg.get("role") == "assistant" and msg.get("content"):
                            level_used = item.get("level_used", user_level)
                            data_samples.append({
                                "text": msg["content"],
                                "label_str": level_used,
                                "source": "instruction"
                            })
    except Exception as e:
        print(f"오류: {e}")

    print(f"Instruction: {len(data_samples)}개 샘플")
    return data_samples


def prepare_dataset(parallel_file, instruction_file, mode="mixed", instruction_ratio=0.3):
    """
    데이터셋 준비

    mode:
    - "parallel_only": Parallel Corpus만 (V2와 동일)
    - "mixed": 두 데이터를 섞어서 학습
    - "instruction_only": Instruction만 (Fine-tuning 시)

    instruction_ratio: mixed 모드에서 instruction 비율 (0.0 ~ 1.0)
    """

    print("\n" + "=" * 60)
    print(f"데이터 로드 모드: {mode}")
    print("=" * 60 + "\n")

    data_samples = []

    # Parallel Corpus 로드
    if mode in ["parallel_only", "mixed"]:
        parallel_samples = load_parallel_corpus(parallel_file)
        data_samples.extend(parallel_samples)

    # Instruction 로드
    if mode in ["instruction_only", "mixed"]:
        instruction_samples = load_instruction_data(instruction_file)

        if mode == "mixed" and instruction_ratio < 1.0:
            # Instruction 비율 조정
            import random
            random.seed(42)
            n_instruction = int(len(instruction_samples) * instruction_ratio)
            instruction_samples = random.sample(instruction_samples, n_instruction)
            print(f"Instruction 샘플 {instruction_ratio * 100}% 사용: {len(instruction_samples)}개")

        data_samples.extend(instruction_samples)

    if not data_samples:
        raise ValueError("데이터가 없습니다!")

    print(f"\n총 {len(data_samples)}개 샘플")

    # Dataset 생성
    dataset = Dataset.from_list(data_samples)

    def map_labels(example):
        return {"label": label2id.get(example["label_str"], -1)}

    dataset = dataset.map(map_labels, remove_columns=["label_str", "source"])
    dataset = dataset.filter(lambda x: x["label"] != -1)

    # 클래스 분포
    labels = dataset["label"]
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\n=== 클래스 분포 ===")
    for label, count in zip(unique_labels, counts):
        print(f"{id2label[label]}: {count} ({count / len(labels) * 100:.1f}%)")

    # 클래스 가중치
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
    print("\n=== 클래스 가중치 ===")
    for label, weight in zip(unique_labels, class_weights):
        print(f"{id2label[label]}: {weight:.3f}")

    # Stratified split
    train_indices, test_indices = train_test_split(
        range(len(dataset)), test_size=0.1, shuffle=True,
        random_state=42, stratify=labels
    )

    dataset_dict = DatasetDict({
        'train': dataset.select(train_indices),
        'test': dataset.select(test_indices)
    })

    print(f"\nTrain: {len(dataset_dict['train'])}")
    print(f"Validation: {len(dataset_dict['test'])}")

    return dataset_dict, class_weights


class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, focal_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights_tensor = torch.tensor(class_weights, dtype=torch.float) if class_weights is not None else None
        # self.loss_fn = FocalLoss(alpha=weights_tensor, gamma=focal_gamma)
        self.loss_fn = OrdinalLabelSmoothingLoss(alpha=weights_tensor, epsilon=0.1)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def plot_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[id2label[i] for i in range(6)],
                yticklabels=[id2label[i] for i in range(6)])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_cefr_router(
        model_name="microsoft/deberta-v3-large",
        output_dir="./cefr_router_v3",
        data_mode="mixed",
        instruction_ratio=0.3,
        use_lora=True,
        focal_gamma=2.0,
        max_length=1024,
        pretrained_model_path=None,
        # LoRA 파라미터
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.15,
        lora_target_modules=None,
        # Learning rate & Scheduler
        learning_rate=2e-5,
        lr_scheduler_type="linear",
        # 실험 이름 (tensorboard)
        experiment_name=None
):
    """
    CEFR 라우터 학습 V3 - Grid Search 지원

    모든 하이퍼파라미터를 커스터마이즈 가능
    """

    # 기본 target modules
    if lora_target_modules is None:
        lora_target_modules = ["query_proj", "key_proj", "value_proj", "dense"]

    # 실험 이름 자동 생성
    if experiment_name is None:
        lr_str = f"{learning_rate:.0e}".replace("e-0", "e-")
        experiment_name = (f"r{lora_r}_a{lora_alpha}_d{int(lora_dropout*100)}_"
                          f"lr{lr_str}_{lr_scheduler_type}_ratio{int(instruction_ratio*100)}")

    # 데이터셋 준비
    # Note: 파일 경로는 실제 환경에 맞게 조정 필요
    datasets, class_weights = prepare_dataset(
        parallel_file="../data/parallel_corpus/train.jsonl",
        instruction_file="../data/instructions/train.jsonl",
        mode=data_mode,
        instruction_ratio=instruction_ratio
    )

    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # 모델 로드
    if pretrained_model_path:
        print(f"\n사전 학습된 모델 로드: {pretrained_model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_path,
            num_labels=6,
            id2label=id2label,
            label2id=label2id,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=6,
            id2label=id2label,
            label2id=label2id,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # LoRA
    if use_lora and not pretrained_model_path:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            inference_mode=False,
            modules_to_save=["classifier", "pooler"]
        )
        model = get_peft_model(model, lora_config)

        print("\n" + "=" * 60)
        print("Configuration:")
        print(f"  LoRA: r={lora_r}, α={lora_alpha}, dropout={lora_dropout}")
        print(f"  LR: {learning_rate}, Scheduler: {lr_scheduler_type}")
        print(f"  Instruction Ratio: {instruction_ratio}")
        print("=" * 60)

        model.print_trainable_parameters()

    # 토크나이징
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length",
                         truncation=True, max_length=max_length)

    tokenized_datasets = datasets.map(tokenize_function, batched=True,
                                      remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 학습 설정
    run_name = f"{experiment_name}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=20 if not pretrained_model_path else 10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=16,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.3,
        max_grad_norm=0.5,

        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,

        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,

        fp16=True,
        gradient_accumulation_steps=8,
        report_to="tensorboard",
        label_smoothing_factor=0.1,
        lr_scheduler_type=lr_scheduler_type,
        dataloader_num_workers=8,
        seed=42,
        optim="adamw_torch_fused",
    )

    # Metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(labels, predictions, average='weighted')

        print("\n=== Classification Report ===")
        print(classification_report(labels, predictions,
                                    target_names=[id2label[i] for i in range(6)],
                                    zero_division=0))

        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plot_confusion_matrix(labels, predictions, cm_path)

        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted
        }

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.001
    )

    # Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
        class_weights=class_weights,
        focal_gamma=focal_gamma
    )

    print("\n" + "=" * 60)
    print("학습 시작...")
    print(f"Experiment: {experiment_name}")
    print("=" * 60 + "\n")

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n=== 최종 평가 ===")
    final_metrics = trainer.evaluate()
    print(f"Experiment: {experiment_name}")
    print(f"Accuracy: {final_metrics['eval_accuracy']:.4f}")
    print(f"F1-macro: {final_metrics['eval_f1_macro']:.4f}")

    return trainer, model, tokenizer, final_metrics


# ====================================================================
# Grid Search 설정
# ====================================================================

# LoRA 설정들
LORA_CONFIGS = {
    "baseline": {
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.15,
        "lora_target_modules": ["query_proj", "key_proj", "value_proj", "dense"],
    },
    "high_rank": {
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.1,
        "lora_target_modules": ["query_proj", "key_proj", "value_proj", "dense"],
    },
    "extended": {
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.1,
        "lora_target_modules": [
            "query_proj", "key_proj", "value_proj", "dense",
            "output.dense", "intermediate.dense"
        ],
    },
    "extreme": {
        "lora_r": 128,
        "lora_alpha": 256,
        "lora_dropout": 0.05,
        "lora_target_modules": [
            "query_proj", "key_proj", "value_proj", "dense",
            "output.dense", "intermediate.dense"
        ],
    }
}

# Learning Rate 설정들
LR_CONFIGS = {
    "lr_2e5": 2e-5,
    "lr_3e5": 3e-5,
    "lr_5e5": 5e-5,
}

# Scheduler 설정들
SCHEDULER_CONFIGS = {
    "linear": "linear",
    "cosine": "cosine",
    "cosine_restarts": "cosine_with_restarts",
}

# Instruction Ratio 설정들
INSTRUCTION_RATIOS = {
    "ratio_30": 0.3,
    "ratio_50": 0.5,
    "ratio_70": 0.7,
    "ratio_90": 0.9,
    "ratio_100": 1.0,
}


def run_grid_search(
    lora_configs=None,
    lr_configs=None,
    scheduler_configs=None,
    instruction_ratios=None,
    mode="mixed"
):
    """
    모든 하이퍼파라미터 조합에 대해 Grid Search 실행

    Parameters:
        lora_configs: dict, LoRA 설정들 (None이면 기본값 사용)
        lr_configs: dict, Learning rate 설정들 (None이면 기본값 사용)
        scheduler_configs: dict, Scheduler 설정들 (None이면 기본값 사용)
        instruction_ratios: dict, Instruction ratio 설정들 (None이면 기본값 사용)
        mode: str, 데이터 모드

    Returns:
        dict, 모든 실험 결과
    """

    # 기본값 설정
    if lora_configs is None:
        lora_configs = LORA_CONFIGS
    if lr_configs is None:
        lr_configs = LR_CONFIGS
    if scheduler_configs is None:
        scheduler_configs = SCHEDULER_CONFIGS
    if instruction_ratios is None:
        instruction_ratios = INSTRUCTION_RATIOS

    # 모든 조합 생성
    combinations = list(product(
        lora_configs.items(),
        lr_configs.items(),
        scheduler_configs.items(),
        instruction_ratios.items()
    ))

    total_experiments = len(combinations)

    print("\n" + "=" * 80)
    print("🔬 Grid Search 시작")
    print("=" * 80)
    print(f"LoRA 설정: {len(lora_configs)}개")
    print(f"Learning Rate: {len(lr_configs)}개")
    print(f"Scheduler: {len(scheduler_configs)}개")
    print(f"Instruction Ratio: {len(instruction_ratios)}개")
    print(f"\n총 실험 개수: {total_experiments}개")
    print("=" * 80 + "\n")

    results = {}

    for idx, (lora_item, lr_item, sched_item, ratio_item) in enumerate(combinations, 1):
        lora_name, lora_config = lora_item
        lr_name, lr_value = lr_item
        sched_name, sched_value = sched_item
        ratio_name, ratio_value = ratio_item

        exp_name = f"{lora_name}_{lr_name}_{sched_name}_{ratio_name}"

        print("\n" + "🔬" * 40)
        print(f"실험 {idx}/{total_experiments}: {exp_name}")
        print(f"  LoRA: {lora_name} (r={lora_config['lora_r']}, α={lora_config['lora_alpha']})")
        print(f"  LR: {lr_value:.0e}, Scheduler: {sched_value}")
        print(f"  Instruction Ratio: {ratio_value}")
        print("🔬" * 40 + "\n")

        output_dir = f"./experiments/{exp_name}"

        try:
            trainer, model, tokenizer, metrics = train_cefr_router(
                output_dir=output_dir,
                data_mode=mode,
                instruction_ratio=ratio_value,
                lora_r=lora_config["lora_r"],
                lora_alpha=lora_config["lora_alpha"],
                lora_dropout=lora_config["lora_dropout"],
                lora_target_modules=lora_config["lora_target_modules"],
                learning_rate=lr_value,
                lr_scheduler_type=sched_value,
                experiment_name=exp_name
            )

            results[exp_name] = {
                "lora": lora_name,
                "lr": lr_name,
                "scheduler": sched_name,
                "ratio": ratio_name,
                "config": {
                    "lora_r": lora_config["lora_r"],
                    "lora_alpha": lora_config["lora_alpha"],
                    "lora_dropout": lora_config["lora_dropout"],
                    "learning_rate": lr_value,
                    "scheduler": sched_value,
                    "instruction_ratio": ratio_value,
                },
                "metrics": metrics,
                "output_dir": output_dir
            }

            print(f"\n✅ {exp_name} 완료!")
            print(f"   Accuracy: {metrics['eval_accuracy']:.4f}")
            print(f"   F1-macro: {metrics['eval_f1_macro']:.4f}")

        except Exception as e:
            print(f"\n❌ {exp_name} 실패: {e}")
            import traceback
            traceback.print_exc()
            results[exp_name] = {"error": str(e)}

    # 결과 요약
    print("\n" + "=" * 120)
    print("실험 결과 요약 (F1-macro 기준 정렬)")
    print("=" * 120)
    print(f"{'Experiment':<40} {'Acc':<8} {'F1':<8} {'LoRA':<15} {'LR':<10} {'Sched':<20} {'Ratio':<8}")
    print("-" * 120)

    # F1-macro 기준으로 정렬
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if "error" not in v],
        key=lambda x: x[1]["metrics"].get("eval_f1_macro", 0),
        reverse=True
    )

    for exp_name, result in sorted_results:
        metrics = result["metrics"]
        config = result["config"]
        acc = metrics.get("eval_accuracy", 0)
        f1 = metrics.get("eval_f1_macro", 0)

        lora_str = f"r{config['lora_r']}/α{config['lora_alpha']}"
        lr_str = f"{config['learning_rate']:.0e}"
        sched_str = config['scheduler']
        ratio_str = f"{int(config['instruction_ratio']*100)}%"

        print(f"{exp_name:<40} {acc:<8.4f} {f1:<8.4f} {lora_str:<15} {lr_str:<10} {sched_str:<20} {ratio_str:<8}")

    # 에러 출력
    errors = [(k, v) for k, v in results.items() if "error" in v]
    if errors:
        print("\n" + "=" * 120)
        print("실패한 실험:")
        for exp_name, result in errors:
            print(f"  {exp_name}: {result['error']}")

    print("=" * 120)

    # 최고 성능 출력
    if sorted_results:
        best_exp, best_result = sorted_results[0]
        print(f"\n🏆 최고 성능: {best_exp}")
        print(f"   Accuracy: {best_result['metrics']['eval_accuracy']:.4f}")
        print(f"   F1-macro: {best_result['metrics']['eval_f1_macro']:.4f}")
        print(f"   Config: {best_result['config']}")

    return results


if __name__ == "__main__":
    # ========================================
    # 실험 모드 선택
    # ========================================

    GRID_SEARCH = True
    SINGLE_EXPERIMENT = False

    # ========================================
    # Grid Search 설정
    # ========================================

    if GRID_SEARCH:
        # 실험할 설정들 선택 (모두 사용하거나 일부만 선택)

        # 예시 1: 모든 조합 테스트 (3 x 3 x 3 x 3 = 81개 실험)
        # results = run_grid_search()

        # 예시 2: 일부 설정만 테스트
        selected_lora = {
            # "baseline": LORA_CONFIGS["baseline"],
            # "high_rank": LORA_CONFIGS["high_rank"],
            # "extended": LORA_CONFIGS["extended"],
            "extreme": LORA_CONFIGS["extreme"]
        }

        selected_lr = {
            # "lr_2e5": LR_CONFIGS["lr_2e5"],
            # "lr_3e5": LR_CONFIGS["lr_3e5"],
            "lr_5e5": LR_CONFIGS["lr_5e5"]
        }

        selected_scheduler = {
            # "linear": SCHEDULER_CONFIGS["linear"],
            "cosine": SCHEDULER_CONFIGS["cosine"],
            # "cosine_restarts": SCHEDULER_CONFIGS["cosine_restarts"]
        }

        selected_ratio = {
            # "ratio_30": INSTRUCTION_RATIOS["ratio_30"],
            # "ratio_50": INSTRUCTION_RATIOS["ratio_50"],
            # "ratio_70": INSTRUCTION_RATIOS["ratio_70"],
            # "ratio_90": INSTRUCTION_RATIOS["ratio_90"],
            "ratio_100": INSTRUCTION_RATIOS["ratio_100"]
        }

        total_exps = len(selected_lora) * len(selected_lr) * len(selected_scheduler) * len(selected_ratio)
        print(f"\n🔥 Grid Search 시작: {total_exps}개 실험\n")
        results = run_grid_search(
            lora_configs=selected_lora,
            lr_configs=selected_lr,
            scheduler_configs=selected_scheduler,
            instruction_ratios=selected_ratio,
            mode="instruction_only"
        )

    # ========================================
    # 단일 실험 설정
    # ========================================

    if SINGLE_EXPERIMENT:
        print(f"\n🔥 단일 실험 실행\n")
        trainer, model, tokenizer, metrics = train_cefr_router(
            output_dir="./cefr_router_single",
            data_mode="mixed",
            instruction_ratio=0.5,
            lora_r=64,
            lora_alpha=128,
            lora_dropout=0.1,
            lora_target_modules=["query_proj", "key_proj", "value_proj", "dense"],
            learning_rate=3e-5,
            lr_scheduler_type="cosine",
            experiment_name="custom_single"
        )
