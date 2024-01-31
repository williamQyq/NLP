# -*- coding: utf-8 -*-
"""Assigment1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QnliS2kIxyWcoBZDHdsrQ8nhmDqiJLy-

##1. Implement BPE Algorithm (4 marks):

Develop a Python implementation of the Byte Pair Encoding (BPE) algorithm, covering key
steps such as learning byte pair merges and encoding/decoding using the learned merge
operations.
"""

from typing import List, Dict
import re
import matplotlib.pyplot as plt

from collections import defaultdict

class BPE():
    def __init__(self, debug=False, vocab=set() ):
        self.vocab = vocab
        # self.bpe_pairs = set()
        self.debug = debug

    def init_corpus(self, data: List[str]):
        """
        Corpus is dict of char tokens to frequency
        """
        corpus = defaultdict(lambda: 0)
        vocab = set()
        #end of word symbol
        vocab.add('_')

        for line in data:
            for word in line.split():
                corpus[' '.join(list(word)) + ' _'] += 1
                vocab.update(list(word))

        return corpus,vocab

    def isStopChar(self,token:str)->bool:
        return bool(re.match('[,.?"\'!;:]+',token))

    def get_pairs(self, corpus: Dict[str, int]):
        pairs = defaultdict(lambda: 0)

        for word, freq in corpus.items():
            tokens = word.split()

            # Pair to frequency, avoiding stop characters
            for i in range(len(tokens) - 1):
              #  if not (self.isStopChar(tokens[i]) or self.isStopChar(tokens[i+1])):
                  pairs[tokens[i], tokens[i + 1]] += freq

        return pairs

    def merge_corpus(self, pair, corpus):
        """
        merge pair appears in the corpus
        """

        corpus_out = {}
        #treat special chars (.*) as normal chars
        bigram = re.escape(' '.join(pair))

        #find pairs allowing spaces for before or at the end
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in corpus:
            word_out = p.sub(''.join(pair), word)
            corpus_out[word_out] = corpus[word]
        return corpus_out


    def save_pair(self, pair, vocab):
        """
        Save merged pair
        """
        vocab.add(''.join(pair))
        # self.bpe_pairs.add(pair)
        return


    def byte_pair_encoding(self, data, k):
        """
        From the given list of strings, merge most frequent adjacent pairs till k times,
        returns vocab, a set of learnt vocab from corpus
        """
        sens = [sentence.strip() + '.' for sentence in data.split('.') if sentence]
        corpus,vocab = self.init_corpus(sens)
        self.merge_freq = []

        for i in range(k):
            pairs = self.get_pairs(corpus)
            if not pairs:
                break

            #most frequent pair
            best = max(pairs, key=pairs.get)

            self.save_pair(best,vocab)

            corpus = self.merge_corpus(best, corpus)

            if self.debug:
                print(f"\n--After {i} iteration")
                print("Most freq pairs: ", best)
                print("Corpus after update: ", corpus)

            self.merge_freq.append((best, pairs[best]))

        # self.corpus = corpus
        self.vocab.update(vocab)

        return self.vocab, corpus


    def decoding(self, tokens:List[chr], vocab)->str:
        """
        decode tokens in sentence and keep merging back to the original sentences using the learned vocab
        """
        if not vocab:
            print("Vocab not learned")
            return

        # Split the data into words, then characters
        decoded = tokens

        is_merged = True
        while is_merged:
            i=0
            is_merged = False

            while i< len(decoded)-1:
                merged = decoded[i]+decoded[i+1]

                if merged in vocab or merged+'_' in vocab:
                    decoded[i] = merged
                    del decoded[i+1]
                    is_merged = True

                else:
                    i+=1

        return decoded

    def visualize_pair_merge_freq(self):
        """
        visualize
        """

        frequencies = [freq for pair, freq in self.merge_freq]
        vocab_size = [i for i, _ in enumerate(self.merge_freq, start=1)]

        plt.figure(figsize=(10, 6))
        plt.bar(vocab_size, frequencies, color='skyblue')
        plt.xlabel('Merged Pairs')
        plt.ylabel('Frequencies')
        plt.title('Frequency of Byte Pair Merges per Iteration')
        plt.xticks(rotation=45)
        plt.show()
