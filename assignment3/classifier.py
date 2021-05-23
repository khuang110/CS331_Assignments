import numpy as np
import pandas as pd
from collections import defaultdict


# class Classifier:
#     def __init__(self, vocab: list, df: pd.DataFrame,
#                  pos_vocab: list, neg_vocab: list, debug=False):
#         """
#         :param vocab: list of all vocab words (pos and neg)
#         :param df:
#         :param pos_vocab:
#         :param neg_vocab:
#         +------------+--------+
#         | sentiment | corpus |
#         +-----------+--------+
#         """
#         if not debug:
#             self._df = df
#             self._vocab = self.vectorizer(vocab)
#             v = [*pos_vocab, *neg_vocab]
#             self._neg_vocab = self.vectorizer(v)
#             self._pos_vocab = self.vectorizer(pos_vocab)
#             self._counts()
#
#     def vocab(self):
#         if hasattr(self, "_vocab") and self._vocab:
#             return self._vocab
#         raise ValueError("Vocab was never initialized... Should not happen")
#
#     def _counts(self):
#         """
#         Initialize counts of positive and negative sentiment.
#         """
#         pos_sum = 0
#         for key, val in self._pos_vocab.items():
#             pos_sum += val
#         neg_sum = 0
#         for key, val in self._neg_vocab.items():
#             neg_sum += val
#         sum = pos_sum + neg_sum
#         self._pos_count = pos_sum
#         self._neg_count = neg_sum
#         self._word_count = sum
#
#     @staticmethod
#     def vectorizer(sentence: list) -> dict:
#         """Feature vectorizor
#
#         :param sentence: array of classes (words).
#             list of all the classes that can appear in a vector.
#         :return: x : counts of words : defautdict
#
#         Examples
#         --------
#         >>> c.vectorizer(["hello", "i", "foo"])
#         {'hello': 1, 'i': 1, 'foo': 1}
#         >>> c.vectorizer(["foo", "foo", "bar"])
#         {'foo': 2, 'bar': 1}
#
#         """
#         x = defaultdict(int)
#         for f in sentence:
#             x[f] += 1
#         return x
#
#     def init_prob_df(self):
#         """
#         Calculate marginal likelihood
#         :return:
#         """
#         pos_sentence_vectors = defaultdict(float)
#         neg_sentence_vectors = defaultdict(float)
#
#         # Calculate Pw(W | C)
#         # for sentiment in [0, 1]:
#         #   sent_vec = []
#
#         for token in self._pos_vocab:
#             # Denom is sum of 1's in sentiment
#             # denom = self._df["sentiment"].sum() + 2.0
#             denom = self._pos_count + 2.0
#             # Get P(W=T | C=T)
#             # for token in self._pos_vocab:
#             # Add count of wordn given its positive
#             if denom < self._pos_vocab[token] + 1.0:
#                 raise ValueError("denom smaller pos_vocab:", denom)
#             # marginal_likelihood = (self._pos_vocab[token] + 1.0) / denom
#             marginal_likelihood = (self._pos_vocab[token] + 1.0) / denom
#             pos_sentence_vectors[token] = marginal_likelihood
#             # pos_sentence_vectors.append(sent_vec)
#         for token in self._neg_vocab:
#
#             # denom = len(self._df.index) - self._df["sentiment"].sum() + 2.0
#             denom = self._neg_count + 2.0
#             # Get P(W=T | C=F)
#             # for token in self._neg_vocab:
#             # Add count of wordn given its positive
#             if denom < self._neg_vocab[token] + 1.0:
#                 raise ValueError("denom smaller neg_vocab:", denom, "neg ", self._neg_vocab[token])
#             marginal_likelihood = (self._neg_vocab[token] + 1.0) / denom
#             neg_sentence_vectors[token] = marginal_likelihood
#             # neg_sentence_vectors.append(sent_vec)
#
#         # Calculate Pc(C), prior probability
#         # p_pos = (self._pos_count / self._word_count)
#         p_pos, p_neg = 0, 0
#         # p_pos = 0
#         # p_neg = self._neg_count / self._word_count
#         print(p_pos)
#         for key, word in pos_sentence_vectors.items():
#             p_pos += word
#         for key, word in neg_sentence_vectors.items():
#             p_neg += word
#             # pos_sentence_vectors = [[x + pos for x in row] for row in pos_sentence_vectors]
#             # neg_sentence_vectors = [[x + neg for x in row] for row in neg_sentence_vectors]
#             # pos_sentence_vectors = [x for x in pos_sentence_vectors]
#             # neg_sentence_vectors = [x for x in neg_sentence_vectors]
#
#         # print(pos_sentence_vectors)
#         print("Pos:", p_pos)
#         print("Neg:", p_neg)
#
#     def train_bn(self):
#         print()


def train_naive_bayes(vocab, documents, classes):
    from math import log
    flatten = lambda x: [ele for lst in x for ele in lst]
    logprior = {}
    loglikelihood = defaultdict(dict)
    big_doc = {}

    def get_logprior(documents, c):
        class_document_count = len([doc for doc in documents if doc[1] == c])
        #print("class doc count:", c, class_document_count)
        total_document_count = len(documents)
        return np.log(class_document_count / total_document_count)

    def get_bigdoc(documents, c):
        # Get words in vocab with sentiment
        return flatten([doc[0] for doc in documents if doc[1] == c])

    for c in classes:
        logprior[c] = get_logprior(documents, c)
        big_doc[c] = get_bigdoc(documents, c)
        words_in_class_count = sum([big_doc[c].count(w) for w in vocab])

        for word in vocab:
            word_count = big_doc[c].count(word)
            loglikelihood[word][c] = np.log((word_count + 1) /
                                            (words_in_class_count + 1))

    return logprior, loglikelihood


def test_naive_bayes(document, logprior, loglikelihood, classes, vocab):
    # Filter words not in the vocab
    document = [word for word in document[0] if word in vocab]
    prob = {}

    for c in classes:
        prob[c] = logprior[c]
        for word in document:
            prob[c] = prob[c] + loglikelihood[word][c]

    return max(prob.keys(), key=(lambda key: prob[key]))


class Classifier:
    def __init__(self, x, y, vocab, corpus: list):
        self._classes = [0, 1]
        self._vocab = vocab
        self.sent_vocab = {1: self.vectorizer(x), 0: self.vectorizer(y)}
        self._corpus = corpus
        self.likelihood = defaultdict(dict)
        self.alpha = {}

    @staticmethod
    def vectorizer(sentence: list) -> dict:
        """Feature vectorizor

        :param sentence: array of classes (words).
            list of all the classes that can appear in a vector.
        :return: x : counts of words : defautdict


        Examples
        --------
        >>> c.vectorizer(["hello", "i", "foo"])
        {'hello': 1, 'i': 1, 'foo': 1}
        >>> c.vectorizer(["foo", "foo", "bar"])
        {'foo': 2, 'bar': 1}

        """
        x = defaultdict(int)
        for f in sentence:
            x[f] += 1
        return x

    def count(self, cls_label=None):
        """
        Get count of class elements
        :param cls_label: class label <1, 0>
        :return: count in given class
        """
        if cls_label is None:
            if hasattr(self, '_total_count'):
                if cls_label is None:
                    return getattr(self, '_total_count')
            setattr(self, '_total_count', len(self._corpus))
            return getattr(self, '_total_count')
        elif cls_label == 1 or cls_label == 0:
            if hasattr(self, '_count' + str(cls_label)):
                return getattr(self, '_count' + str(cls_label))
            setattr(self, '_count' + str(cls_label),
                    len([s for s in self._corpus if s[1] == cls_label]))
            return getattr(self, '_count' + str(cls_label))
        else:
            raise ValueError("Invalid class label!!")

    def word_count(self, cls_label):
        if cls_label == 0 or cls_label == 1:
            if hasattr(self, '_word_count'+str(cls_label)):
                return getattr(self, '_word_count'+str(cls_label))
            setattr(self, '_word_count'+str(cls_label),
                    sum(self.flatten_words(cls_label).count(w) for w in self._vocab))
            return getattr(self, '_word_count'+str(cls_label))

    def flatten_words(self, cls_label):
        return list(np.concatenate([w[0] for w in self._corpus if w[1] == cls_label]))

    def train_naive_bayes(self):
        vocab_word_count = self.vectorizer(self._vocab)

        for cls_label in self._classes:
            self.alpha[cls_label] = np.log(self.count(cls_label) / self.count())

            for word in self._vocab:
                word_count = self.flatten_words(cls_label).count(word)
                self.likelihood[word][cls_label] = np.log((word_count + 1) /
                                                (self.word_count(cls_label) + 1))

    def test_naive_bayes(self, sentence):
        """
        Test the naive bayes net
        :param sentence: Sentence to test
        :return: Class prediction
        """
        if not self.alpha or not self.likelihood:
            raise ValueError("Must train bayes net before testing")

        sentence = [word for word in sentence[0] if word in self._vocab]
        prediction = {}

        for cls_label in self._classes:
            prediction[cls_label] = self.alpha[cls_label]
            for word in sentence:
                prediction[cls_label] = prediction[cls_label] + self.likelihood[word][cls_label]

        return max(prediction, key=lambda x: prediction[x])
