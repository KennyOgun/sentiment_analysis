# sentiment_analysis
This repo is created to perform sentiment analysis on cutsomer's product feedback.

Sentiment analysis can be approached in several ways depending on the data, goals, and resources available. Here's a comprehensive list of algorithms and methods used for sentiment analysis, categorized into:

Rule-Based Methods

Machine Learning-Based Methods

Deep Learning-Based Methods

Hybrid Methods

Foundation Models / Transformers

🔹 1. Rule-Based Sentiment Analysis
These rely on manually crafted rules like lexicons, part-of-speech tagging, and syntactic parsing.

Algorithm / Method	Description
VADER (Valence Aware Dictionary for Sentiment Reasoning)	Lexicon and rule-based sentiment analysis tool tuned for social media texts.
TextBlob	Simple NLP tool with rule-based polarity scoring.
SentiWordNet	Uses WordNet synsets to score sentiment based on pre-tagged lexicons.
LIWC (Linguistic Inquiry and Word Count)	Psycholinguistic tool that categorizes words into psychologically meaningful categories.
Afinn	Assigns sentiment scores to words (-5 to +5) using a dictionary.
🟦 Category: Rule-Based

🔹 2. Classical Machine Learning Algorithms
These use labeled data and statistical features (like TF-IDF, BoW).

Algorithm	Description
Naive Bayes (NB)	Probabilistic classifier, simple and efficient for text.
Logistic Regression	Binary/multiclass classifier based on a linear decision boundary.
Support Vector Machines (SVM)	Finds hyperplane to separate sentiment classes.
Random Forest	Ensemble tree-based classifier, often robust to overfitting.
K-Nearest Neighbors (KNN)	Classifies based on nearest similar examples.
Gradient Boosting (XGBoost, LightGBM)	Boosted decision trees for higher accuracy.
🟦 Category: Machine Learning-Based

🔹 3. Deep Learning Methods
These are data-hungry but powerful for capturing semantic and contextual sentiment.

Algorithm	Description
Convolutional Neural Networks (CNN)	Detects sentiment using n-gram features (spatial patterns).
Recurrent Neural Networks (RNN)	Captures sequence info, useful for context.
LSTM (Long Short-Term Memory)	Variant of RNN that handles long-term dependencies.
GRU (Gated Recurrent Unit)	Simplified LSTM with fewer parameters.
BiLSTM / BiGRU	Bidirectional versions for better context awareness.
🟦 Category: Deep Learning-Based

🔹 4. Hybrid Methods
Combine rule-based + ML or deep learning approaches.

Method	Description
VADER + ML	Use VADER scores as features in ML model.
Lexicon + SVM/LogReg	Combine sentiment lexicon outputs with statistical models.
Feature Engineering + Deep Learning	Manually engineered sentiment features + LSTM/RNN.
🟦 Category: Hybrid

🔹 5. Transformer-Based / Foundation Models
These are pre-trained on massive corpora and fine-tuned for sentiment tasks.

Model	Description
BERT (Bidirectional Encoder Representations from Transformers)	Pre-trained language model fine-tuned for sentiment.
RoBERTa	Robustly optimized version of BERT.
DistilBERT	Lightweight, faster BERT version for real-time sentiment.
XLNet	Improves upon BERT by capturing permutation-based context.
ALBERT	Lighter and scalable version of BERT.
T5 / GPT-3 / ChatGPT / LLaMA	Large generative models that can handle zero-shot or few-shot sentiment tasks.
BART	Seq2Seq transformer good for summarization and classification.
DeBERTa	Enhanced attention mechanisms for better performance.
🟦 Category: Transformer-Based / Foundation Models

🧠 Bonus: Feature Extraction Techniques (used in ML/DL)
Bag of Words (BoW)

TF-IDF

Word2Vec

GloVe

FastText

BERT embeddings

✅ Summary Table
Category	Techniques / Models
Rule-Based	VADER, TextBlob, SentiWordNet, LIWC, Afinn
Machine Learning	Naive Bayes, Logistic Regression, SVM, Random Forest, XGBoost
Deep Learning	CNN, RNN, LSTM, GRU, BiLSTM, BiGRU
Hybrid	VADER + ML, Lexicon + SVM, Custom Features + LSTM
Transformer Models	BERT, RoBERTa, DistilBERT, XLNet, GPT, T5, ALBERT, BART, DeBERTa




# binary_sentiment+prediction notebook
I applied the rules based, machine learning and deep learning  model to binary sentiment analysis on amazon review using Kaggle Kernel

https://www.kaggle.com/code/kennyolowu/binary-sentiment-prediction/edit
