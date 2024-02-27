# Word Embedding, CBOW, and Skip-gram Report

**Yuqing Qiao**  
02/25/2024

## Description

Word embeddings are a class of techniques where individual words are encoded as vectors of values within a space defined by the vocabulary size. Each word is mapped to a unique vector, and the values of these vectors are learned in a neural network. This allows words sharing common semantics to have similar representations in the vector space.

### CBOW

The Continuous Bag of Words (CBOW) model predicts a target word from a bag of surrounding context words. The input layer represents the context words, which are averaged before being passed to a single hidden layer, in a manner similar to the Skip-gram model. The output layer is a softmax layer that predicts the target word.

![CBOW training Snapshot](/word%20embedding/snapshot/cbowSnapshot.png)

### Skip-gram

The Skip-gram model architecture involves taking a target word and predicting the surrounding context words. For each target word, it examines a window of surrounding context words and attempts to predict them based on the target word itself. The process involves an embedding layer, which is then flattened before being passed to a softmax layer for prediction.

![Skip-gram training Snapshot](/word%20embedding/snapshot/skipgramSnapshot.png)

## Dataset and Pre-processing Steps

The dataset for training these models is a large corpus of text data. Pre-processing steps include tokenizing the text into words, removing stop words and punctuation, converting all words to lowercase, and generating context-target pairs. It is optional to transform words into one-hot vectors for input into the models, thanks to the use of an embedding layer and the TensorFlow framework.

## Evaluation Results

The CBOW model successfully identifies the similarity of related words using word embeddings, such as how “paris” is to “rome” as “china” is to “japan”. Quantitative measures like cosine similarity between vectors are used and the similarity of word vector embeddings is visualized using PCA. However, the training of the Skip-gram model is more memory-consuming compared to CBOW.

![PCA visualization](/word%20embedding/snapshot/PCA.png)

![Cosine similarity](/word%20embedding/snapshot/cosineSimilarity.png)

## Challenges and Potential Improvements

Implementing these models comes with challenges, including handling large text files, which can lead to memory constraints and slow training times. Dealing with rare words and capturing the nuances of language can also be difficult. I have made several improvements, including batch processing, using ProcessPoolExecutor to process data, employing negative sampling for efficient training (specifically for Skip-gram), and adjusting hyperparameters such as the context window size and learning rate for more efficient training.

## Conclusion

The development of word embeddings, particularly through models like Skip-gram and CBOW, has been transformative for NLP, enabling a range of applications from sentiment analysis to machine translation. By effectively capturing the semantic relationships between words, these models provide a foundation for understanding and processing language in a nuanced and meaningful way.
