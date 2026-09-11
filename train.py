import os
import json
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)


# =========================================================
# 1. CONFIGURATION
# =========================================================

MODEL_NAME = "distilbert-base-uncased"

DATA_PATH = "data/intents.csv"

OUTPUT_DIR = "models/distilbert-intent"

MAX_LENGTH = 128


# =========================================================
# 2. CHECK DEVICE
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("DEVICE:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Training on CPU")
    print("CPU training may take longer.")

print("=" * 60)


# =========================================================
# 3. LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

print("\nDataset:")
print(df.head())

print("\nNumber of rows:", len(df))

print("\nIntent distribution:")
print(df["intent"].value_counts())


# =========================================================
# 4. CLEAN DATA
# =========================================================

df = df.dropna(subset=["text", "intent"])

df["text"] = df["text"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()

df = df[df["text"] != ""]


# =========================================================
# 5. CREATE LABEL MAPPING
# =========================================================

labels = sorted(df["intent"].unique())

label2id = {
    label: i
    for i, label in enumerate(labels)
}

id2label = {
    i: label
    for label, i in label2id.items()
}

df["label"] = df["intent"].map(label2id)

print("\nLabels:")

for label, index in label2id.items():
    print(index, "=", label)


# =========================================================
# 6. TRAIN / VALIDATION / TEST SPLIT
# =========================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# For this small demonstration dataset, we use
# the test set for evaluation.
# Once the dataset is larger, we will create a
# separate validation set.

validation_df = test_df.copy()

print("\nDataset split:")
print("Training:", len(train_df))
print("Validation:", len(validation_df))
print("Testing:", len(test_df))

# =========================================================
# 7. CONVERT TO HUGGING FACE DATASETS
# =========================================================

train_dataset = Dataset.from_pandas(
    train_df[["text", "label"]],
    preserve_index=False
)

validation_dataset = Dataset.from_pandas(
    validation_df[["text", "label"]],
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df[["text", "label"]],
    preserve_index=False
)


# =========================================================
# 8. LOAD DISTILBERT TOKENIZER
# =========================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# =========================================================
# 9. TOKENIZATION
# =========================================================

def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH
    )


train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

validation_dataset = validation_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


# =========================================================
# 10. LOAD DISTILBERT MODEL
# =========================================================

print("\nLoading DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)


# =========================================================
# 11. DATA COLLATOR
# =========================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# =========================================================
# 12. EVALUATION METRICS
# =========================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# =========================================================
# 13. TRAINING CONFIGURATION
# =========================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    # CPU training
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    num_train_epochs=3,

    weight_decay=0.01,

    # Show training progress
    logging_steps=1,

    # Disable external logging
    report_to="none",

    # CPU does not use FP16
    fp16=False,

    # Keep the best model
    load_best_model_at_end=True,

    # Select best model using F1 score
    metric_for_best_model="f1",

    greater_is_better=True
)


# =========================================================
# 14. TRAINER
# =========================================================

trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)


# =========================================================
# 15. TRAIN
# =========================================================

print("\nStarting training...\n")

trainer.train()


# =========================================================
# 16. TEST MODEL
# =========================================================

print("\nEvaluating on test dataset...\n")

test_results = trainer.evaluate(
    test_dataset
)

print("\nTEST RESULTS")
print("=" * 60)

for key, value in test_results.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.4f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


# =========================================================
# 17. SAVE MODEL
# =========================================================

print("\nSaving model...")

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


# =========================================================
# 18. SAVE LABEL MAPPING
# =========================================================

with open(
    os.path.join(OUTPUT_DIR, "labels.json"),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "label2id": label2id,
            "id2label": {
                str(k): v
                for k, v in id2label.items()
            }
        },
        f,
        indent=4
    )


print("\nTraining completed successfully.")

print("Model saved at:")
print(OUTPUT_DIR)