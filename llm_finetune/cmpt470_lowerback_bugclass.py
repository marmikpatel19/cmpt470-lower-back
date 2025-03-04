import pandas as pd
import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
# Suppress Python warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Load dataset
df = pd.read_csv("Combined_Classifications.csv")
df.dropna(subset=["title", "description", "classification"], inplace=True)

# Ensure text and classification labels are clean
df["title"] = df["title"].astype(str)
df["description"] = df["description"].astype(str)
df["classification"] = df["classification"].astype(str)

# Balance dataset by oversampling
max_class_count = df["classification"].value_counts().max()

balanced_dfs = []
for class_label in df["classification"].unique():
    class_subset = df[df["classification"] == class_label]
    
    # Oversample minority classes to match the largest class
    balanced_subset = resample(class_subset, replace=True, n_samples=max_class_count, random_state=SEED)
    
    balanced_dfs.append(balanced_subset)

# Combine all balanced subsets and shuffle
balanced_df = pd.concat(balanced_dfs).sample(frac=1, random_state=SEED)

# Encode text and labels
balanced_df["text"] = balanced_df["title"] + " " + balanced_df["description"]
label_encoder = LabelEncoder()
balanced_df["label"] = label_encoder.fit_transform(balanced_df["classification"])

# Split dataset: 80% Train, 10% Validation, 10% Test
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    balanced_df["text"].tolist(), balanced_df["label"].tolist(), test_size=0.2, random_state=SEED, stratify=balanced_df["label"]
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=SEED, stratify=temp_labels
)

# Choose model: CodeBERT, DistilBERT, or GraphCodeBERT
MODEL_NAME = "microsoft/codebert-base"  # Replace with "distilbert-base-uncased" or "microsoft/graphcodebert-base"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# Ensure all text values are strings and not None
train_texts = [str(text) for text in train_texts]
val_texts = [str(text) for text in val_texts]
test_texts = [str(text) for text in test_texts]

# Tokenize the data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

# Create a PyTorch dataset class
class BugReportDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# Convert tokenized data into Dataset objects
train_dataset = BugReportDataset(train_encodings, train_labels)
val_dataset = BugReportDataset(val_encodings, val_labels)
test_dataset = BugReportDataset(test_encodings, test_labels)

# Load Pretrained Model for Classification
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(label_encoder.classes_)
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./class_results",  # Save model here
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",  # Save checkpoints at the end of each epoch
    save_total_limit=2,  # Keep only the best and the last model
    load_best_model_at_end=True,  # Load best model at the end
    metric_for_best_model="accuracy",  # Define best model criterion
    learning_rate=2e-5,  # Use lower LR
    greater_is_better=True,  # Higher accuracy is better
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./class_logs",
    logging_steps=10,
    push_to_hub=False,
)
# create folder to store tokenizer
os.makedirs("./class_results/tokenizer", exist_ok=True)

# Define compute metrics
def compute_metrics(pred):
    predictions = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()
tokenizer.save_pretrained("class_results/tokenizer")


# Evaluate on test set
test_results = trainer.evaluate(test_dataset)
print("Test Results:", test_results)

# Evaluate on test set and get predictions
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)  # Convert logits to class predictions
actual_labels = predictions.label_ids  # Get actual labels

# Convert numerical labels back to text labels
actual_labels_text = label_encoder.inverse_transform(actual_labels)
predicted_labels_text = label_encoder.inverse_transform(preds)

# Create a DataFrame with actual vs. predicted results
test_results_df = pd.DataFrame({
    "input_text" : test_texts,
    "actual_label": actual_labels_text,
    "predicted_label": predicted_labels_text
})

test_results_df.to_csv("test_output.csv", index=False)
