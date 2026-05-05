# NLP Word2Vec & NER Pipeline

This project implements a complete Natural Language Processing (NLP) pipeline combining:

- Word embedding using **Word2Vec (Skip-Gram + Negative Sampling)**
- Named Entity Recognition (NER) using:
  - Feed-Forward Neural Network (FFNN)
  - Hidden Markov Model (HMM)

---

## Overview

The project is divided into two main parts:

### Part 1: Word Embeddings
- Train Word2Vec model from scratch
- Skip-Gram architecture
- Negative Sampling optimization
- Word analogy tasks (e.g. king → queen)

### Part 2: Named Entity Recognition
Two different approaches:

#### 1. Feed-Forward Neural Network (FFNN)
- Uses pretrained Word2Vec embeddings
- Context window feature extraction
- BatchNorm, Dropout, and optimization techniques

#### 2. Hidden Markov Model (HMM)
- Transition probabilities
- Emission probabilities
- Viterbi decoding
- Laplace smoothing + UNK handling

---

## Project Structure


.
├── word2vec_ner_pipeline.ipynb # Full implementation
├── word_embeddings.npy # Saved embeddings
├── vocab.pkl # Vocabulary mappings
├── epoch_losses.png # Training visualization
└── README.md


---

## Dataset

- Dataset used: **CoNLL-2003**
- Loaded via HuggingFace:

lhoestq/conll2003


Includes:
- Tokens
- Named Entity tags (PER, ORG, LOC, etc.)

---

## Features

- Text preprocessing and cleaning
- Vocabulary construction with frequency filtering
- Negative sampling optimization
- Training visualization (loss curves)
- Word analogy testing
- NER feature engineering using context windows
- Model comparison (HMM vs Neural Network)

---

## Concepts Covered

- Word2Vec (Skip-Gram)
- Negative Sampling
- Embedding spaces & cosine similarity
- Named Entity Recognition (NER)
- Hidden Markov Models (HMM)
- Viterbi Algorithm
- Feed-Forward Neural Networks
- Class imbalance handling

---

## How to Run

1. Clone the repo:
```bash
git clone https://github.com/ShamsNabih03/nlp-word2vec-ner-models.git

