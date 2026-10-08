"""
학습된 CEFR 분류기로 텍스트의 레벨을 예측합니다.

사용 예:
    python predict.py "The cat sat on the mat."
    python predict.py --model ./results --file texts.txt   # 한 줄에 텍스트 하나
"""

import argparse
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main(argv=None):
    parser = argparse.ArgumentParser(description="CEFR 레벨 예측")
    parser.add_argument("texts", nargs="*", help="예측할 텍스트")
    parser.add_argument("--model", default="dksysd/cefr-classifier", help="HF 모델 ID 또는 로컬 경로")
    parser.add_argument("--file", default=None, help="한 줄에 텍스트 하나씩 들어있는 파일 ('-'는 stdin)")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--all_scores", action="store_true", help="모든 레벨의 확률 출력")
    args = parser.parse_args(argv)

    texts = list(args.texts)
    if args.file:
        stream = sys.stdin if args.file == "-" else open(args.file, encoding="utf-8")
        with stream:
            texts.extend(line.strip() for line in stream if line.strip())
    if not texts:
        parser.error("텍스트를 인자로 주거나 --file 을 지정하세요.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()
    id2label = model.config.id2label

    for start in range(0, len(texts), args.batch_size):
        batch = texts[start:start + args.batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=args.max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1).cpu()

        for text, p in zip(batch, probs):
            idx = int(p.argmax())
            line = f"{id2label[idx]}\t{p[idx]:.3f}\t{text}"
            if args.all_scores:
                line += "\t" + " ".join(f"{id2label[i]}={v:.3f}" for i, v in enumerate(p.tolist()))
            print(line)


if __name__ == "__main__":
    main()
