# Fine-Tuning LLMs for Automated Bug Classification

## Link For Different Milestone Documents:

M2: https://drive.google.com/file/d/1BHX8DCHmhQtVhIAj1AZ5nQEZMiEOOq-o/view?usp=sharing

Inter-Rater Agreement Analysis: https://drive.google.com/file/d/1Fxg03apcc_Jz2k5z7zOTeQT9UJ7fYiFd/view?usp=sharing

Percentage for Each Classification: https://drive.google.com/file/d/1EnEoQtHoG3CQIa3LdbShx1tBglua1HsG/view?usp=sharing

Project Proposal: https://drive.google.com/file/d/14XIKnZEHcgoxKSqagBHJRKMTdTnBo38q/view?usp=sharing

Model Evaluation Report: Coming Soon

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

