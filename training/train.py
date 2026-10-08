"""
CEFR Router V3 - 학습 및 Grid Search

모든 하이퍼파라미터 조합을 자동으로 실험:
- LoRA 파라미터 (rank, alpha, dropout, target_modules)
- Learning rate
- LR Scheduler
- Instruction ratio

사용 예:
    python training/train.py --mode mixed --lr 3e-5 --output_dir ./results
    python training/train.py --grid_search --grid_lora baseline,high_rank --grid_lr lr_2e5
"""

import argparse
import os
from itertools import product

import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
import torch
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    cohen_kappa_score, mean_absolute_error
)
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from loss import OrdinalLabelSmoothingLoss, FocalLoss
    from data import (
        DEFAULT_INSTRUCTION_FILE, DEFAULT_PARALLEL_FILE,
        group_train_test_split, id2label, label2id, load_samples
    )
except ImportError:
    # Fallback for when running from root directory
    from training.loss import OrdinalLabelSmoothingLoss, FocalLoss
    from training.data import (
        DEFAULT_INSTRUCTION_FILE, DEFAULT_PARALLEL_FILE,
        group_train_test_split, id2label, label2id, load_samples
    )

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def prepare_dataset(parallel_file, instruction_file, mode="mixed", instruction_ratio=0.3, test_size=0.1):
    """
    데이터셋 준비

    mode:
    - "parallel_only": Parallel Corpus만
    - "mixed": 두 데이터를 섞어서 학습
    - "instruction_only": Instruction만 (Fine-tuning 시)

    instruction_ratio: mixed 모드에서 instruction 비율 (0.0 ~ 1.0)

    같은 원문/질문에서 나온 레벨별 텍스트는 train/test 중 한쪽에만 들어갑니다 (group split).
    """

    print("\n" + "=" * 60)
    print(f"데이터 로드 모드: {mode}")
    print("=" * 60 + "\n")

    data_samples = load_samples(parallel_file, instruction_file, mode=mode,
                                instruction_ratio=instruction_ratio)
    print(f"\n총 {len(data_samples)}개 샘플")

    labels = [label2id[s["label_str"]] for s in data_samples]
    groups = [s["group"] for s in data_samples]

    # 클래스 분포
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\n=== 클래스 분포 ===")
    for label, count in zip(unique_labels, counts):
        print(f"{id2label[label]}: {count} ({count / len(labels) * 100:.1f}%)")

    # Group split (원문/질문 단위)
    train_indices, test_indices = group_train_test_split(labels, groups, test_size=test_size, seed=42)

    # 클래스 가중치 (train split 기준)
    train_labels = [labels[i] for i in train_indices]
    class_weights = compute_class_weight('balanced', classes=np.arange(len(label2id)), y=train_labels)
    print("\n=== 클래스 가중치 ===")
    for label, weight in enumerate(class_weights):
        print(f"{id2label[label]}: {weight:.3f}")

    dataset = Dataset.from_dict({
        "text": [s["text"] for s in data_samples],
        "label": labels,
    })

    dataset_dict = DatasetDict({
        'train': dataset.select(train_indices),
        'test': dataset.select(test_indices)
    })

    print(f"\nTrain: {len(dataset_dict['train'])} ({len({groups[i] for i in train_indices})} groups)")
    print(f"Validation: {len(dataset_dict['test'])} ({len({groups[i] for i in test_indices})} groups)")

    return dataset_dict, class_weights


class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, loss_type="ols", focal_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights_tensor = torch.tensor(class_weights, dtype=torch.float) if class_weights is not None else None
        if loss_type == "focal":
            self.loss_fn = FocalLoss(alpha=weights_tensor, gamma=focal_gamma)
        else:
            self.loss_fn = OrdinalLabelSmoothingLoss(alpha=weights_tensor, epsilon=0.1)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def ordinal_metrics(labels, predictions):
    """레벨 간 서열을 고려한 지표"""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    return {
        "qwk": cohen_kappa_score(labels, predictions, weights="quadratic"),
        "adjacent_accuracy": float(np.mean(np.abs(labels - predictions) <= 1)),
        "mae": mean_absolute_error(labels, predictions),
    }


def plot_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))
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
        parallel_file=DEFAULT_PARALLEL_FILE,
        instruction_file=DEFAULT_INSTRUCTION_FILE,
        data_mode="mixed",
        instruction_ratio=0.3,
        use_lora=True,
        loss_type="ols",
        focal_gamma=2.0,
        max_length=1024,
        num_train_epochs=None,
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

    os.makedirs(output_dir, exist_ok=True)

    # 데이터셋 준비
    datasets, class_weights = prepare_dataset(
        parallel_file=parallel_file,
        instruction_file=instruction_file,
        mode=data_mode,
        instruction_ratio=instruction_ratio
    )

    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # 모델 로드
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_path or model_name,
        num_labels=6,
        id2label=id2label,
        label2id=label2id,
    )
    if pretrained_model_path:
        print(f"\n사전 학습된 모델 로드: {pretrained_model_path}")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
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

    # 토크나이징 (패딩은 collator가 배치 단위로 처리)
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    tokenized_datasets = datasets.map(tokenize_function, batched=True,
                                      remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if num_train_epochs is None:
        num_train_epochs = 20 if not pretrained_model_path else 10

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=experiment_name,
        num_train_epochs=num_train_epochs,
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

        fp16=use_cuda,
        gradient_accumulation_steps=8,
        report_to="tensorboard",
        lr_scheduler_type=lr_scheduler_type,
        dataloader_num_workers=8,
        seed=42,
        optim="adamw_torch_fused" if use_cuda else "adamw_torch",
    )

    # Metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        print("\n=== Classification Report ===")
        print(classification_report(labels, predictions,
                                    labels=list(range(6)),
                                    target_names=[id2label[i] for i in range(6)],
                                    zero_division=0))

        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plot_confusion_matrix(labels, predictions, cm_path)

        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1_macro": f1_score(labels, predictions, average='macro', zero_division=0),
            "f1_weighted": f1_score(labels, predictions, average='weighted', zero_division=0),
            **ordinal_metrics(labels, predictions),
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
        loss_type=loss_type,
        focal_gamma=focal_gamma
    )

    print("\n" + "=" * 60)
    print("학습 시작...")
    print(f"Experiment: {experiment_name}")
    print("=" * 60 + "\n")

    trainer.train()

    print("\n=== 최종 평가 ===")
    final_metrics = trainer.evaluate()
    print(f"Experiment: {experiment_name}")
    for key in ("accuracy", "f1_macro", "qwk", "adjacent_accuracy", "mae"):
        print(f"{key}: {final_metrics[f'eval_{key}']:.4f}")

    # LoRA 어댑터를 병합하여 transformers만으로 로드 가능한 모델로 저장
    final_model = trainer.model
    if isinstance(final_model, PeftModel):
        final_model = final_model.merge_and_unload()
    final_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"모델 저장: {output_dir}")

    return trainer, final_model, tokenizer, final_metrics


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
    mode="mixed",
    output_root="./experiments",
    **train_kwargs
):
    """
    모든 하이퍼파라미터 조합에 대해 Grid Search 실행

    Parameters:
        lora_configs: dict, LoRA 설정들 (None이면 기본값 사용)
        lr_configs: dict, Learning rate 설정들 (None이면 기본값 사용)
        scheduler_configs: dict, Scheduler 설정들 (None이면 기본값 사용)
        instruction_ratios: dict, Instruction ratio 설정들 (None이면 기본값 사용)
        mode: str, 데이터 모드
        output_root: str, 실험별 결과가 저장될 상위 디렉토리
        train_kwargs: train_cefr_router에 그대로 전달되는 추가 인자

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

        output_dir = os.path.join(output_root, exp_name)

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
                experiment_name=exp_name,
                **train_kwargs
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
            print(f"   QWK: {metrics['eval_qwk']:.4f}")

        except Exception as e:
            print(f"\n❌ {exp_name} 실패: {e}")
            import traceback
            traceback.print_exc()
            results[exp_name] = {"error": str(e)}

    # 결과 요약
    print("\n" + "=" * 128)
    print("실험 결과 요약 (F1-macro 기준 정렬)")
    print("=" * 128)
    print(f"{'Experiment':<40} {'Acc':<8} {'F1':<8} {'QWK':<8} {'LoRA':<15} {'LR':<10} {'Sched':<20} {'Ratio':<8}")
    print("-" * 128)

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
        qwk = metrics.get("eval_qwk", 0)

        lora_str = f"r{config['lora_r']}/α{config['lora_alpha']}"
        lr_str = f"{config['learning_rate']:.0e}"
        sched_str = config['scheduler']
        ratio_str = f"{int(config['instruction_ratio']*100)}%"

        print(f"{exp_name:<40} {acc:<8.4f} {f1:<8.4f} {qwk:<8.4f} {lora_str:<15} {lr_str:<10} {sched_str:<20} {ratio_str:<8}")

    # 에러 출력
    errors = [(k, v) for k, v in results.items() if "error" in v]
    if errors:
        print("\n" + "=" * 128)
        print("실패한 실험:")
        for exp_name, result in errors:
            print(f"  {exp_name}: {result['error']}")

    print("=" * 128)

    # 최고 성능 출력
    if sorted_results:
        best_exp, best_result = sorted_results[0]
        print(f"\n🏆 최고 성능: {best_exp}")
        print(f"   Accuracy: {best_result['metrics']['eval_accuracy']:.4f}")
        print(f"   F1-macro: {best_result['metrics']['eval_f1_macro']:.4f}")
        print(f"   Config: {best_result['config']}")

    return results


def _select(configs, keys, name):
    """콤마로 구분된 키 목록으로 설정 딕셔너리를 필터링"""
    if not keys:
        return configs
    selected = {}
    for key in keys.split(","):
        key = key.strip()
        if key not in configs:
            raise SystemExit(f"--grid_{name}: 알 수 없는 키 '{key}' (가능: {', '.join(configs)})")
        selected[key] = configs[key]
    return selected


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CEFR 분류기 학습 / Grid Search")
    parser.add_argument("--parallel_file", default=str(DEFAULT_PARALLEL_FILE))
    parser.add_argument("--instruction_file", default=str(DEFAULT_INSTRUCTION_FILE))
    parser.add_argument("--output_dir", default="./results",
                        help="단일 실험의 저장 위치, Grid Search에서는 실험별 하위 디렉토리의 상위 경로")
    parser.add_argument("--mode", default="mixed", choices=["mixed", "parallel_only", "instruction_only"])
    parser.add_argument("--instruction_ratio", type=float, default=0.5)
    parser.add_argument("--model_name", default="microsoft/deberta-v3-large")
    parser.add_argument("--pretrained_model_path", default=None,
                        help="이미 학습된 모델에서 이어서 학습 (LoRA 미적용)")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--loss", default="ols", choices=["ols", "focal"])
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    single = parser.add_argument_group("단일 실험")
    single.add_argument("--lr", type=float, default=3e-5)
    single.add_argument("--scheduler", default="cosine")
    single.add_argument("--no_lora", action="store_true", help="LoRA 없이 전체 fine-tuning")
    single.add_argument("--lora_r", type=int, default=64)
    single.add_argument("--lora_alpha", type=int, default=128)
    single.add_argument("--lora_dropout", type=float, default=0.1)
    single.add_argument("--experiment_name", default=None)

    grid = parser.add_argument_group("Grid Search")
    grid.add_argument("--grid_search", action="store_true")
    grid.add_argument("--grid_lora", default=None, help=f"예: baseline,extreme (가능: {', '.join(LORA_CONFIGS)})")
    grid.add_argument("--grid_lr", default=None, help=f"가능: {', '.join(LR_CONFIGS)}")
    grid.add_argument("--grid_scheduler", default=None, help=f"가능: {', '.join(SCHEDULER_CONFIGS)}")
    grid.add_argument("--grid_ratio", default=None, help=f"가능: {', '.join(INSTRUCTION_RATIOS)}")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    common = dict(
        model_name=args.model_name,
        parallel_file=args.parallel_file,
        instruction_file=args.instruction_file,
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        loss_type=args.loss,
        focal_gamma=args.focal_gamma,
    )

    if args.grid_search:
        run_grid_search(
            lora_configs=_select(LORA_CONFIGS, args.grid_lora, "lora"),
            lr_configs=_select(LR_CONFIGS, args.grid_lr, "lr"),
            scheduler_configs=_select(SCHEDULER_CONFIGS, args.grid_scheduler, "scheduler"),
            instruction_ratios=_select(INSTRUCTION_RATIOS, args.grid_ratio, "ratio"),
            mode=args.mode,
            output_root=args.output_dir,
            **common
        )
    else:
        train_cefr_router(
            output_dir=args.output_dir,
            data_mode=args.mode,
            instruction_ratio=args.instruction_ratio,
            use_lora=not args.no_lora,
            pretrained_model_path=args.pretrained_model_path,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            learning_rate=args.lr,
            lr_scheduler_type=args.scheduler,
            experiment_name=args.experiment_name,
            **common
        )


if __name__ == "__main__":
    main()
