# NLP && LLM
author: Yuqing Qiao  
date: June 2024

## Table of Contents
- [Introduction](#introduction)
- [Byte Pair Encoding (BPE) Tokenization](#byte-pair-encoding-bpe-tokenization)
- [Sentiment Analysis on Movie Reviews](#sentiment-analysis-on-movie-reviews)
  - [Classifiers Implementation](#classifiers-implementation)
  - [Visualization of Classifiers](#visualization-of-classifiers)
- [Emotion Detection with TensorFlow](#emotion-detection-with-tensorflow)
- [Word Embedding Models](#word-embedding-models)
  - [Skip-gram and Continuous Bag of Words (CBOW)](#skip-gram-and-continuous-bag-of-words-cbow)
  - [Visualization of Word Embeddings](#visualization-of-word-embeddings)
- [LLM](#llm)
  - [BERT](#bert)
  - [Transformer](#transformer)

## Introduction
This project aims to implement several natural language processing (NLP) techniques from scratch, including Byte Pair Encoding (BPE) for tokenization, sentiment analysis with multiple classifiers, emotion detection using TensorFlow, and word embedding models like Skip-gram and CBOW. I also explore the visualization of classifier decisions and word embeddings to understand the underlying patterns in data.

## Byte Pair Encoding (BPE) Tokenization
Byte Pair Encoding (BPE) is a subword tokenization technique that iteratively merges the most frequent pair of bytes or characters in a given text. This approach helps in handling out-of-vocabulary words more effectively.
- [BPE Implementation Folder](bpe/BPE.ipynb)

## Sentiment Analysis on Movie Reviews
Performed sentiment analysis on movie reviews, aiming to classify them as positive or negative using Naive Bayes (NB), Logistic Regression (LR), and Multi-Layer Perceptron (MLP) classifiers and compare their performance on the metrics of accuracy, precision, f1-score and recall.

### Classifiers Implementation
- Naive Bayes, Logistic Regression, and Multi-Layer Perceptron (MLP) classifiers are implemented to predict the sentiment of movie reviews.
- [Sentiment Analysis Implementation Folder](/sentimental%20analysis/Sentiment_Analysis.ipynb)

### Visualization of Classifiers
- We use t-SNE to visualize the decision boundaries of NB, LR, and MLP classifiers to understand how they separate different topics from all articles.
- [Classifiers Visualization Folder](/classifier/ArticleClassifier&EmotionPrediction.ipynb)

## Emotion Detection with TensorFlow
We implement term frequency (TF) and term frequency-inverse document frequency (TF-IDF) from scratch and use them to train a TensorFlow model for classifying six different emotions in corpus.
- [Emotion Detection Implementation Folder](/classifier/ArticleClassifier&EmotionPrediction.ipynb)

## Word Embedding Models
We explore word embeddings through self-implemented Skip-gram and Continuous Bag of Words (CBOW) models, focusing on capturing semantic relationships between words.

### Skip-gram and Continuous Bag of Words (CBOW)
- [Word Embedding Models Folder](/word%20embedding/Skipgram&CBOW.ipynb)

### Visualization of Word Embeddings
- We visualize word embeddings using PCA and cosine similarity to understand the semantic relationships of word embedding captured by the Skip-gram and CBOW models.
- [Word Embeddings Visualization Folder](/word%20embedding/Skipgram&CBOW.ipynb)

## LLM

### BERT
- [BERT Folder](/BERT/BERT.ipynb)

### Transformer
- [Transformer Folder](/Transformer/PA2_Yuqing_(Transformer).ipynb)
