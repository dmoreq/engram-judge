#!/usr/bin/env python3
"""Fine-tune ModernBERT / Laya Cross-Encoder on BigCloneBench using Kaggle GPU T4.

Fully compatible with Transformers 5.x and 4.x.
Outputs:
  - Checkpoint in `./output_laya_bcb/best_model` (SafeTensors + config)
  - Detailed Evaluation Metrics (F1, Precision, Recall, Accuracy)
  - Zipped file `laya_bcb_model.zip` ready for download.
"""

import os
import sys

# ==============================================================================
# 0. CÀI ĐẶT / CẬP NHẬT THƯ VIỆN TRƯỚC KHI IMPORT (BẮT BUỘC TRÊN KAGGLE)
# ==============================================================================
os.system("pip install -q -U transformers accelerate datasets scikit-learn safetensors")

import gc
import json
import inspect
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ==============================================================================
# 1. CẤU HÌNH & THAM SỐ (TỐI ƯU CHO GPU T4 16GB)
# ==============================================================================
# "answerdotai/ModernBERT-base" (149M - Rất nhẹ, train cực nhanh, khuyến nghị)
# "convaiinnovations/laya" hoặc "answerdotai/ModernBERT-large" (421M - Nặng hơn)
MODEL_NAME = "answerdotai/ModernBERT-base"

# Số mẫu huấn luyện (Dataset BigCloneBench có >900k mẫu; lấy 50,000 mẫu là đủ
# để hội tụ F1 > 0.95 trong ~1.5 giờ trên GPU T4)
MAX_TRAIN_SAMPLES = 50_000
MAX_VAL_SAMPLES = 4_000
MAX_TEST_SAMPLES = 1_000

MAX_SEQ_LENGTH = 512
BATCH_SIZE = 16          # T4 16GB chạy batch_size 16 cực kỳ an toàn
GRAD_ACCUM_STEPS = 2     # Effective batch size = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 2
OUTPUT_DIR = "./output_laya_bcb"

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--> Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# ==============================================================================
# 2. NẠP VÀ TIỀN XỬ LÝ DỮ LIỆU BIGCLONEBENCH
# ==============================================================================
print("--> Loading CodeXGLUE BigCloneBench dataset from HuggingFace...")
dataset = load_dataset("google/code_x_glue_cc_clone_detection_big_clone_bench")

train_data = dataset["train"].select(range(min(len(dataset["train"]), MAX_TRAIN_SAMPLES)))
val_data = dataset["validation"].select(range(min(len(dataset["validation"]), MAX_VAL_SAMPLES)))
test_data = dataset["test"].select(range(min(len(dataset["test"]), MAX_TEST_SAMPLES)))

print(f"--> Dataset splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    # Cross-encoder: ghép cặp [CLS] func1 [SEP] func2 [SEP]
    result = tokenizer(
        examples["func1"],
        examples["func2"],
        truncation="longest_first",
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )
    # Khóa chặt cột nhãn là 'labels' kiểu int 0 hoặc 1 (chuẩn int64 của CrossEntropyLoss)
    result["labels"] = [1 if bool(lbl) else 0 for lbl in examples["label"]]
    return result

print("--> Tokenizing datasets...")
train_encoded = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names)
val_encoded = val_data.map(preprocess_function, batched=True, remove_columns=val_data.column_names)
test_encoded = test_data.map(preprocess_function, batched=True, remove_columns=test_data.column_names)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

# ==============================================================================
# 3. HÀM ĐÁNH GIÁ CHỈ SỐ (F1, PRECISION, RECALL, ACCURACY)
# ==============================================================================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# ==============================================================================
# 4. KHỞI TẠO MÔ HÌNH VÀ THIẾT LẬP TRAINING ARGUMENTS
# ==============================================================================
print(f"--> Initializing Cross-Encoder Model: {MODEL_NAME}...")
# BẮT BUỘC: Khóa cứng problem_type="single_label_classification" để ModernBERT
# dùng CrossEntropyLoss (nhận labels 1D size [16]), thay vì tự suy đoán nhầm
# sang BCEWithLogitsLoss (đòi hỏi nhãn 2D size [16, 2]).
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    problem_type="single_label_classification",
    id2label={0: "NON_CLONE", 1: "CLONE"},
    label2id={"NON_CLONE": 0, "CLONE": 1}
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    warmup_steps=300,
    fp16=True,                             # Tiết kiệm VRAM và tăng tốc trên T4
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none"
)

# Tự động tương thích cả Transformers 5.x (processing_class) và 4.x (tokenizer)
trainer_kwargs = {
    "model": model,
    "args": training_args,
    "train_dataset": train_encoded,
    "eval_dataset": val_encoded,
    "data_collator": data_collator,
    "compute_metrics": compute_metrics,
}
if "processing_class" in inspect.signature(Trainer.__init__).parameters:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = Trainer(**trainer_kwargs)

# ==============================================================================
# 5. HUẤN LUYỆN MÔ HÌNH
# ==============================================================================
print("--> Starting Training...")
trainer.train()

# ==============================================================================
# 6. ĐÁNH GIÁ CUỐI CÙNG TRÊN TEST SPLIT
# ==============================================================================
print("\n" + "=" * 60)
print("--> RUNNING FINAL EVALUATION ON TEST SET...")
print("=" * 60)

test_results = trainer.evaluate(test_encoded)
print("\n[RESULT TABLE] BigCloneBench Test Performance:")
print(f"  * Test F1 Score:   {test_results['eval_f1']:.4f}")
print(f"  * Test Precision:  {test_results['eval_precision']:.4f}")
print(f"  * Test Recall:     {test_results['eval_recall']:.4f}")
print(f"  * Test Accuracy:   {test_results['eval_accuracy']:.4f}")
print("=" * 60)

# ==============================================================================
# 7. LƯU MÔ HÌNH VÀ TẠO FILE ZIP TẢI VỀ
# ==============================================================================
best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
trainer.save_model(best_model_dir)
tokenizer.save_pretrained(best_model_dir)

os.system(f"zip -r laya_bcb_model.zip {best_model_dir}")
print(f"\n--> Model saved and zipped successfully: `laya_bcb_model.zip`")
print(f"--> File ready for download in Kaggle working directory.")
