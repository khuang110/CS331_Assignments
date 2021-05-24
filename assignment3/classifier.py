import numpy as np
from collections import defaultdict


class Classifier:
    def __init__(self, x, y, vocab, corpus: list):
        self._classes = [0, 1]
        self._vocab = vocab
        self.sent_vocab = {1: self.vectorizer(x), 0: self.vectorizer(y)}
        self._corpus = corpus
        self.conditional_probability = defaultdict(dict)
        self.prior_probability = {}

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

    def tokenize_corpus(self):
        return self.vectorizer(self.flatten_words(0) + self.flatten_words(1))

    def update_priors(self, y_true, y_pred):
        prior_idx = []
        for i, (t, p) in enumerate(zip(y_true, y_pred)):
            if t != p:
                prior_idx.append(i)
        for i in prior_idx:
            for sentence in self._corpus[i][0]:
                for word in sentence[0]:
                    if word in self.prior_probability.keys():
                        self.prior_probability[word][self._corpus[i][1]] += prior_idx

    def train_naive_bayes(self):
        for cls_label in self._classes:
            normal_prior = np.log(self.count(cls_label) / self.count())

            for word in self._vocab:
                #self.prior_probability[word][cls_label] = normal_prior
                self.prior_probability[cls_label] = normal_prior
                word_count = self.flatten_words(cls_label).count(word)
                self.conditional_probability[word][cls_label] = np.log((word_count + 1) /
                                                                       (self.word_count(cls_label) + 1))

    def test_naive_bayes(self, sentence):
        """
        Test the naive bayes net
        :param sentence: Sentence to test
        :return: Class prediction
        """
        if not self.prior_probability or not self.conditional_probability:
            raise ValueError("Must train bayes net before testing")

        sentence = [word for word in sentence[0] if word in self._vocab]
        prediction = {}

        for cls_label in self._classes:
            prediction[cls_label] = 0
            for word in sentence:
                prediction[cls_label] += self.conditional_probability[word][cls_label]

        return 0 if prediction[0] > prediction[1] else 1


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """
    s = 1. / (1. + np.exp(-1. * x))


    return s

