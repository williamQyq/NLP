# 1.1 BPE Algorithm Implementation

## Overview
This project implements the Byte Pair Encoding (BPE) algorithm in Python. BPE is a subword tokenization method originally designed for data compression and adapted for Natural Language Processing. This implementation focuses on learning byte pair merges and encoding/decoding using the learned operations.

`The Python file contains only the implementation of the BPE class. For testing, plotting graphs, and displaying details of the BPE algorithm, please refer to the BPE.ipynb notebook.`

## Features
- Initializes the corpus and generates vocabulary
- Identifies and merges frequent byte pairs
- Customizable number of merge operations (`k`)
- Visualization of merge frequency and vocabulary growth
- Debug mode for detailed iteration output

## Dependencies
- matplotlib for visualization

To install matplotlib, run:

```bash
pip install matplotlib
```

## Usage

Create vocab, corpus
```bash
from bpe import BPE

bpe = BPE(debug=True)

data = "your training data"
k = 100  # Number of merge operations
vocab, corpus = bpe.byte_pair_encoding(data, k)
```

Decoding
```bash

new_data = "data to encode"
tokens = [char for word in new_data.split() for char in word]
decoded = bpe.decoding(tokens, vocab)
```


# 1.2 Text Classification Assignment: Movie Review Sentiment Analysis

## Overview
This assignment's primary goal is to implement and evaluate three text classification algorithms—Naive Bayes, Logistic Regression, and Multilayer Perceptron (MLP)—on the NLTK Movie Reviews dataset. The focus is on exploring the impact of two different feature representations: raw Term Frequency (TF) and Term Frequency-Inverse Document Frequency (TF-IDF).

## Dependency

```bash 
pip install nltk scikit-learn matplotlib
```

`Assignment1_2_sentiment_analysis.ipynb" contains the visualization of the classifier evaluation.`