# 📚 NLP Analysis & Classification of Narrative Texts  
**Classical NLP × Machine Learning × Transformers**

## 🔍 Project Overview
This project presents an **end-to-end Natural Language Processing (NLP) system** for analyzing, classifying, and interpreting narrative text using a combination of **linguistic analysis, classical machine learning models, and Transformer-based architectures**.

The goal is to achieve **strong predictive performance** while also understanding how **linguistic structure, writing style, sentiment, and emotion** influence narrative outcomes. The project demonstrates how modern NLP pipelines are **designed, evaluated, and interpreted** in real-world settings.

## 🎯 Objectives
- Analyze syntactic and linguistic patterns across different writing styles  
- Classify story settings using **Word2Vec + SVM** and **BERT**
- Predict story outcomes using **sentiment analysis**
- Examine emotional patterns across narrative themes
- Compare **classical NLP approaches vs Transformer models**
- Apply **hyperparameter tuning** to improve model performance


## 🧠 NLP Tasks Covered

### Linguistic & Structural Analysis
- Tokenization, POS tagging, and dependency parsing using **spaCy**
- Measurement of syntactic complexity via **dependency span analysis**
- Statistical comparison of POS tag usage across:
  - Writing styles
  - Story outcomes (victory vs defeat)

**Key Insight:**  
Descriptive and poetic styles exhibit significantly higher syntactic complexity, while concise and children’s writing favors simpler structures.


### Text Classification (Story Setting Prediction)
Two modeling approaches were implemented and compared:

#### 🔹 Classical Approach
- **Word2Vec embeddings** (Google News, 300-dimensional)
- **Support Vector Machine (SVM)**
- Hyperparameter tuning using **RandomizedSearchCV**

**Accuracy improved from 80% → 91% after optimization**

#### 🔹 Transformer-Based Approach
- **Fine-tuned BERT** (`bert-base-uncased`)
- Context-aware sequence classification

**Final accuracy: 95.5%**, outperforming classical methods, especially on semantically overlapping classes.

**Key Insight:**  
Transformer models capture contextual and narrative nuances more effectively than feature-based embeddings.


### Sentiment Analysis (Outcome Prediction)
- Fine-tuned BERT model trained on the **final sentence of each story**
- Predicts whether a story ends in **victory or defeat**
- Compared:
  - Baseline BERT
  - Hyperparameter-optimized BERT (using **Optuna**)

**Trade-off observed:**  
Improved precision at the cost of recall — reflecting real-world evaluation considerations.

### Emotion Analysis
- Emotion classification using a **pre-trained RoBERTa-based emotion model**
- Emotion distribution analyzed across narrative themes:
  - Rebellion
  - Discovery
  - Betrayal
  - Love
  - Redemption

**Key Insights:**
- Fear and sadness dominate darker themes (rebellion, betrayal)
- Joy appears more prominently in discovery and redemption narratives


## 📊 Evaluation & Analysis
- Accuracy, Precision, Recall, F1-score
- Confusion matrices and pairplots
- Statistical hypothesis testing (t-tests)
- Visual analysis of embeddings and logits

The project emphasizes **interpretability and diagnostic analysis**, not just raw performance metrics.

## 🛠️ Tech Stack
- **Python**
- **spaCy** - for linguistic processing
- **NLTK** - for text preprocessing
- **Gensim** - for Word2Vec embeddings
- **Scikit-learn** - for SVM, evaluation, and tuning
- **PyTorch** - for model training
- **Hugging Face Transformers** - for BERT and emotion models
- **Optuna** - for hyperparameter optimization
- **Matplotlib / Seaborn / Plotly** - for visualization


## 🚀 Real-World Applications
The techniques demonstrated in this project are applicable to:
- Document and content classification
- Narrative and discourse analysis
- Customer feedback and review mining
- Editorial and content analytics
- Sentiment-driven decision systems
- NLP-powered content intelligence tools
