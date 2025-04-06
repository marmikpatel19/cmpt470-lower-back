# Fine-Tuning LLMs for Automated Bug Classification

## Link For Different Milestone Documents:

Milestone 2: https://drive.google.com/file/d/1BHX8DCHmhQtVhIAj1AZ5nQEZMiEOOq-o/view?usp=sharing

Inter-Rater Agreement Analysis: https://drive.google.com/file/d/1Fxg03apcc_Jz2k5z7zOTeQT9UJ7fYiFd/view?usp=sharing

Percentage for Each Classification: https://drive.google.com/file/d/1EnEoQtHoG3CQIa3LdbShx1tBglua1HsG/view?usp=sharing

Fined Tunded LLMs full Data: https://usaskca1-my.sharepoint.com/:f:/g/personal/ara258_usask_ca/EtB5Mw2O7vFAs4_zz2WVHysB0CHA1tHKSd4NgLkdYz_mxQ?e=okqAzw

Models Evaluation Report: https://drive.google.com/file/d/19Nbt6afPuo6z1ajquKTGEa-b8xR_dilv/view?usp=sharing

Project Proposal for Milestone 3: https://drive.google.com/file/d/14XIKnZEHcgoxKSqagBHJRKMTdTnBo38q/view?usp=sharing

Project Proposal for Milestone 4: https://drive.google.com/file/d/1fJm1nMtnN8eNsXVWWNbrWF4GLfi6RaF6/view?usp=sharing

Project Proposal for Milestone 5: https://drive.google.com/file/d/190VINoBtoqZ6mKo0-mKzQtg2F1L3OTfA/view?usp=sharing

Presentation Slides: https://docs.google.com/presentation/d/1UArFkzsltQq3Azejfe2cvDD90rAO6j08/edit?usp=sharing&ouid=116311615735165561564&rtpof=true&sd=true

Paper: https://drive.google.com/file/d/1-EZ82nrDkz-cz7pluI41sm9CC6QkuIQV/view?usp=sharing

OverLeaf LaTex Paper link: https://www.overleaf.com/1636111949nbfvbfrkycjg#91c3c8

##

## Project Overview

**Course:** CMPT 470 - Advanced Software Engineering
**Group Name:** Lower-Back

Bug classification is a critical aspect of software maintenance, yet many bug-tracking platforms, such as GitHub, provide only a general "bug" label. This lack of structured classification leads to inefficiencies, requiring developers to manually review and categorize bugs, increasing development time and costs. This project explores the potential of fine-tuning **large language models (LLMs)** to automate the bug classification process and improve software maintenance workflows.

## Problem Statement

GitHub’s current issue-tracking system lacks structured bug classification, leading to inefficiencies in bug resolution. Developers must manually categorize bug reports, which is time-consuming and inconsistent. By fine-tuning **transformer-based models**, we aim to automate this classification process, reducing developer workload and improving software development efficiency.

## Objectives

- Develop a **fine-tuned LLM** model for automated bug classification.
- Create a high-quality, labeled dataset of **GitHub bug reports**.
- Evaluate the performance of transformer-based models in bug classification.
- Identify patterns in bug types across different repositories.

## Research Questions

1. How accurately can fine-tuned LLMs classify bug reports into predefined categories?
2. Which bug types (e.g., syntax errors, performance issues) are more challenging to classify?
3. What trends can be observed in bug distribution across different software repositories?

## Methodology

### Dataset

- Data Source: **GitHub Issues API** (Public repositories)
- Filtering Criteria:
    - Issues labeled as "bug"
    - Exclude feature requests and unrelated discussions
    - Include diverse repositories (AI, Frontend, Tools, etc.)
    - Focus on open issues for analysis
- Target Size: **1,000 – 2,000 bug reports** from the last 3–5 years

### Selected Repositories and their matching Links

- **React** ⇒ [https://github.com/facebook/react/issues?q=is%3Aissue state%3Aopen label%3A"Type%3A Bug"](https://github.com/facebook/react/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22Type%3A%20Bug%22)
- **VS Code** ⇒ [https://github.com/microsoft/vscode/issues?q=is%3Aissue state%3Aopen label%3Abug](https://github.com/microsoft/vscode/issues?q=is%3Aissue%20state%3Aopen%20label%3Abug)
- **Scikit-learn** ⇒ [https://github.com/scikit-learn/scikit-learn/issues?q=is%3Aissue state%3Aopen label%3ABug](https://github.com/scikit-learn/scikit-learn/issues?q=is%3Aissue%20state%3Aopen%20label%3ABug)
- **TensorFlow** ⇒ [https://github.com/tensorflow/tensorflow/issues?q=is%3Aissue state%3Aopen label%3Atype%3Abug](https://github.com/tensorflow/tensorflow/issues?q=is%3Aissue%20state%3Aopen%20label%3Atype%3Abug)

### Labeling Process

Each bug report will be classified into one of the following categories:

- **Syntax Error**
- **Runtime Error**
- **Performance Issue**
- **Security Vulnerability**
- **Logical Bug**
- **Dependency Issue**
- **UI/UX Bug**

We will perform **inter-rater agreement analysis** to ensure consistency in classification. If agreement falls below **60%**, we will refine classification categories.

### Model Training & Evaluation

- **Baseline models:** Naïve Bayes, SVM
- **Fine-tuned transformer models:** CodeBERT, CodeT5
- **Evaluation metrics:** Precision, Recall, F1-score
- **Feature Engineering:**
    - Text-based features (bug descriptions, comments)
    - Code snippet analysis for improved classification

## Expected Outcomes

- A fine-tuned LLM capable of accurately classifying bug reports.
- A labeled dataset of GitHub bug reports.
- Insights into common bug types across repositories.
- Improved issue-tracking and bug triage systems for developers.

### Instructions for running the llm_finetune and traditional_ml_training

For all the below make sure you clone the project first using git clone.

For using the llm_finetune model go to directory called llm_finetune 

For using the traditional_ml_training go to directory called traditional_ml_training

# Model Training and Evaluation Guide llm_finetune

## Prerequisites
Before running the model, ensure you have the following:
- Python 3.8 or later
- An internet connection (to download pretrained models from Hugging Face)
- (Optional) A GPU for faster training (CUDA support recommended)

## Installation
### Step 1: Set Up a Virtual Environment (Recommended)
It is recommended to create a virtual environment to manage dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### Step 2: Install Required Packages
Run the following command to install all dependencies:

```bash
pip install pandas numpy scikit-learn torch transformers datasets
```

If you have a **GPU**, install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
(Modify the CUDA version based on your system, or install `cpuonly` if you don’t have a GPU.)

### Step 3: Verify Installation
Run the following command to check if PyTorch is installed correctly:
```python
import torch
print(torch.cuda.is_available())  # Should return True if GPU is available
```

## Running the Model
### Step 1: Prepare the Dataset
Ensure that you have a CSV file named `Combined_Classifications.csv` with the required columns:
- `title` (Title of the text)
- `description` (Detailed description)
- `classification` (Label for classification)

Place this file in the same directory as the script.

### Step 2: Run the Training Script
Run the following command to train and evaluate the model:
```bash
python train_model.py
```

This script will:
1. Load and preprocess the dataset
2. Tokenize the text using a pretrained model
3. Train a classifier on the dataset
4. Evaluate the model on a test set
5. Save results to `test_output.csv`

### Step 3: View Results
Once the model finishes training, you can check the predictions in the output file:
```bash
cat test_output.csv  # View results on macOS/Linux
type test_output.csv  # View results on Windows
```

## Troubleshooting
- If the model runs slowly, check if CUDA is enabled by running `torch.cuda.is_available()`.
- If you run into tokenization errors, ensure `transformers` is up to date:
  ```bash
  pip install --upgrade transformers
  ```

## Notes
- The script uses `microsoft/codebert-base` by default. You can change this to another model like `distilbert-base-uncased`.
- The training results and logs are stored in `./class_results`.
- If needed, modify `train_model.py` to adjust training parameters.



# Traditional Machine Learning Model Training Guide

## Prerequisites
Before running the model, ensure you have the following:
- Python 3.8 or later
- An internet connection (to install dependencies)
- (Optional) A GPU for faster processing (recommended for large datasets)

## Installation
### Step 1: Set Up a Virtual Environment (Recommended)
It is recommended to create a virtual environment to manage dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### Step 2: Install Required Packages
Run the following command to install all dependencies:

```bash
pip install pandas numpy scikit-learn
```

### Step 3: Verify Installation
Run the following command to check if all required libraries are installed:
```python
import pandas as pd
import numpy as np
import sklearn
print("All dependencies are installed correctly.")
```

## Running the Model
### Step 1: Prepare the Dataset
Ensure that you have a CSV file named `Combined_Classifications.csv` with the required columns:
- `title` (Title of the text)
- `description` (Detailed description)
- `classification` (Label for classification)

Place this file in the same directory as the script.

### Step 2: Run the Training Script
Run the following command to train and evaluate the model:
```bash
python train_traditional_ml.py
```

This script will:
1. Load and preprocess the dataset
2. Tokenize and vectorize text using TF-IDF
3. Train either a **Naïve Bayes** or **SVM** model (configurable in script)
4. Evaluate the model on a test set
5. Save results to `traditional_ml_test_output.csv`

### Step 3: View Results
Once the model finishes training, you can check the predictions in the output file:
```bash
cat traditional_ml_test_output.csv  # View results on macOS/Linux
type traditional_ml_test_output.csv  # View results on Windows
```

## Choosing a Model
The script allows you to choose between **Naïve Bayes** and **SVM** classifiers:
- **Naïve Bayes**: Works well for text classification with smaller datasets.
- **SVM**: Provides better performance for complex text classification tasks.

To change the model type, modify the following line in `train_traditional_ml.py`:
```python
MODEL_TYPE = "Naive Bayes"  # Change to "SVM" if needed
```

## Troubleshooting
- If you run into TF-IDF vectorization issues, make sure `Combined_Classifications.csv` contains non-empty text values.

## Notes
- The script balances the dataset through oversampling to ensure fair training.
- The training results and logs are displayed in the console.
- If needed, modify `train_traditional_ml.py` to adjust hyperparameters.


## Hypothesis

- Certain bug types, particularly **security issues** and **performance issues**, may be harder to classify due to limited training data.
- Integrating **code snippets** into classification models will improve accuracy.
- Fine-tuned LLMs will **outperform traditional ML models** (Naïve Bayes, SVM) in bug classification tasks.

## Implications

- Enhancing **issue-tracking** platforms like GitHub.
- Reducing **developer workload** by automating bug triage.
- Advancing **AI-assisted debugging tools**.

## Contributors

- **Ardalan Askarian** (ara258)
- **Princess Tayab** (prt898)
- **Timofei Kabakov** (tik981)
- **Marmik Patel** (qay871)

