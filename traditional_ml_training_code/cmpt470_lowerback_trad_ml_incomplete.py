import pandas as pd
import random
import numpy as np
import os
import warnings
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

## TODO: import library for vectorization, naive bayes and svm classifier

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Suppress warnings
warnings.filterwarnings("ignore")

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

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

# TODO: Convert text data into numerical representation
"""
Use vectorization to convert train_texts, val_texts and test_texts from string to number.

Some suggestions: TF-IDF vectorization, CountVectorizer
"""
# Write your code here



# TODO: Choose model: Naive Bayes or SVM
"""Work with Naive Bayes and SVM Classifier"""
# Write your code here



# TODO: Train the model
# Write your code here



# TODO: Evaluate on validation set
# Write your code here




# Compute accuracy and classification report for validation set
val_accuracy = accuracy_score(val_labels, val_preds)
val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average="weighted")

print(f"\nModel: {MODEL_TYPE}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-score: {val_f1:.4f}")


# TODO: Evaluate on test set
# Write your code here



# Compute accuracy and classification report for test set
test_accuracy = accuracy_score(test_labels, test_preds)
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_labels, test_preds, average="weighted")

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-score: {test_f1:.4f}")

# Convert numerical labels back to text labels
actual_labels_text = label_encoder.inverse_transform(test_labels)
predicted_labels_text = label_encoder.inverse_transform(test_preds)

# Save predictions
test_results_df = pd.DataFrame({
    "input_text": test_texts,
    "actual_label": actual_labels_text,
    "predicted_label": predicted_labels_text
})
test_results_df.to_csv("traditional_ml_test_output.csv", index=False)


"""
Resources to complete the code:

1. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
2. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
3. https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
4. https://scikit-learn.org/stable/modules/svm.html

"""
