import pandas as pd
import numpy as np
import heapq
import math
import sys, os
import operator as op
from functools import reduce


class NaiveBayes:
    def __init__(self, *, p=None, smoothing=1e-9):
        self.classes = None
        self.smoothing = smoothing
        self.prior = p

    @property
    def classes(self):
        return self._classes

    @classes.setter
    def classes(self, separated_classes):
        self._classes = separated_classes

    @staticmethod
    def _sep_by_class(dataset):
        sep = {}
        for i in range(len(dataset)):
            v = dataset[i]
            if v[-1] not in sep:
                sep[v[-1]].append(v)
        return sep

    def fit(self, X, y, sample_weight=None):
        return self._partial_fit(X, y, np.unique(y), refit=True,
                                 sample_weight=sample_weight)

    def _partial_fit(self, X, y,  classes_=None, sample_weight=None,
                                refit=False):
        if refit:
            self.classes = None

        self.epsilon = self.smoothing * np.var(X, axis=0).max()

        n_classes = len(self.classes)
        n_feats = X.shape(1)
        self.theta_ = np.zeros(n_feats)

    @staticmethod
    def summarize(dataset):
        summaries = [(np.mean(attr), np.std(attr)) for attr in zip(*dataset)]
        del summaries[-1]
        return summaries

    def summarize_by_class(self, dataset):

        summaries = {}
        for sep in dataset:
            for classVal, inst in sep.items():
                summaries[classVal] = self.summarize(inst)
        return summaries

    @staticmethod
    def calc_probability(x, mean, stdev):
        exp = np.exp(-(np.power(x - mean, 2) / (2 * np.power(stdev, 2))))
        return (1 / (np.sqrt(2 * np.pi) * stdev)) * exp

    def calc_class_probability(self, summaries, input_vector):
        probs = {}
        for classVal, classSummaries in summaries.items():
            probs[classVal] = 1
            for i in range(len(classSummaries)):
                mean, stdev = classSummaries[i]
                x = input_vector[i]
                probs[classVal] *= self.calc_probability(x, mean, stdev)
        return probs

    def predict(self, summaries, input_vector):
        probs = self.calc_class_probability(summaries, input_vector)
        best_label, best_prob = None, -1

        for classVal, prob in probs.items():
            if best_label is None or prob > best_prob:
                best_label = classVal
                best_prob = prob
        return best_label

    def get_predictions(self, summaries, test_set):
        predictions = []

        for i in range(len(test_set)):
            res = self.predict(summaries, test_set[i])
            predictions.append(res)
        return predictions

    @staticmethod
    def accuracy(test_set, predictions):
        correct = 0
        for i in range(len(test_set)):
            if test_set[i][-1] == predictions[i]:
                correct += 1
        return (correct / float(len(test_set))) * 100.0


def vectorizer(classes: list) -> dict:
    """  Feature vectorizor

    :param classes: array of classes (words).
        list of all the classes that can appear in a vector.
    :return: x : list

    Examples
    --------
    >>> vectorizer(["hello", "i", "foo"])
    {"hello": 1, "i": 1, "foo": 1}
    >>> vectorizer(["foo", 'foo', 'bar'])
    {"foo": 2 'bar': 1}
    """
    x = {}
    for f in classes:
        if f in x:
            x[f] += 1
        else:
            x[f] = 1
    return x


class Classifier:
    def __init__(self, classes, df: pd.DataFrame):
        self.classes_ = classes
        self.df = df
        word_freq = vectorizer(classes)
        #self.most_freq = heapq.nlargest(200, word_freq, key=word_freq.get)
        self.most_freq = word_freq
        self._total_classes = 0
        self._pos_count = 0
        self._neg_count = 0

        # arrays containing rows of words
        self._sum_words = []
        self._sentence_vectors = []

    def _get_counts(self):
        counts = self.df['sentiment'].value_counts().to_dict()
        self._total_classes = len(self.df.index)
        self._pos_count = counts[1]
        self._neg_count = counts[0]

    def total_classes(self):
        if hasattr(self, "_total_count") and self._total_classes:
            return self._total_classes

        self._get_counts()
        return self._total_classes

    def pos_count(self):
        if hasattr(self, "_pos_count") and self._pos_count:
            return self._pos_count

        self._get_counts()
        return self._pos_count

    def neg_count(self):
        if hasattr(self, "_neg_count") and self._pos_count:
            return self._pos_count

        self._get_counts()
        return self._neg_count

    @staticmethod
    def sigmoid(x):
        """
            Compute the sigmoid
        :param x: scalar or array
        :return: sigmoid
        """
        return 1.0 / (1.0 + np.exp(-1.0 * x))

    def p_class(self, class_id=None) -> float:
        """
            probability of class type
            P(Class=<1,0>)
        :return:
        """
        if class_id is None:
            raise ValueError("Must enter class_id!")

        if class_id == 1:
            return self.pos_count() / self.total_classes()
        elif class_id == 0:
            return self.neg_count() / self.total_classes()
        else:
            raise ValueError("Invalid class_id")

    def sum_words(self, class_id=None):
        """
            Sum of words given some class id
            n
            Σ  (Class_id=<1,0> ^ wordn)
            i=0
        :param class_id:
        :return:
        """
        if class_id is None:
            raise ValueError("Must enter a class_id!")

        if not hasattr(self, "_sentence_vectors") and not self._sentence_vectors:
            self.bag_words()

        #self._sum_words = self._sentence_vectors.sum(axis=0, keepdims=True)
        vocab_sum = []
        for idx, row in enumerate(self._sentence_vectors):
            # Check if class id and sum

            if class_id == self.df['sentiment'].iloc[idx]:
                #print("idx", idx, " sentiment idx:", self.df['sentiment'].iloc[idx])
                sum = 0
                for j in row:
                    sum += j
                vocab_sum.append(sum)
        return vocab_sum

    def marginal_likelihood(self, prior, y):
        alpha, beta = prior
        h = np.sum(y)
        ln = len(y)
        p_y = np.exp(binomial(alpha + h, beta + n - h) - binomial(alpha, beta))
        return p_y

    def uniform_dirchlet_priors(self):
        """
                              (# records with Xj=uj and Y=v) +1
            P(Xj=uj | Y=v) = ------------------------------------
                                   (# records with Y=v) + Nj
            where:
                Nj = # values that Xj can take on. class=<1,0> := Nj=2

        :return:
        +--------+--------+-----+------+
        | Class  | word1 | ... | wordn |
        +--------+-------+----+--------+
        | <1,0> |
        """


    def bag_words(self):
        """

        :param df_doc: dataframe of corpus
            +-----------+-------------+
            | Sentiment  |   corpus   |
            +-----------+------------+
            | < 0, 1>   | "sentence" |
            +-----------+------------+

        :return: sentence_vectors : matrix
            row = sentence
            col = word
            +-----------+---------+-------+------+
            | Sentence  |  word  |  word |  ...
            +-----------+--------+-------+------+
        """

        try:
            sentence_vectors = []
            for sentence in self.df["corpus"]:
                sentence_toks = sentence.rsplit(sep=" ")
                sent_vec = []

                for tok in self.most_freq:
                    if tok in sentence_toks:
                        sent_vec.append(1)
                    else:
                        sent_vec.append(0)
                self._sentence_vectors.append(sent_vec)
            return self._sentence_vectors
        except:
            catch_err()

    def train_naive_bayes(self):
        """
            Predict for each sentence.
            Ypred = argmax P(Class=v | Word1=u1 ... Wordn=un)
                      v
        :return: Ypred
        """

def cons(n, r):
    r = min(r, n - r)
    numer = 1
    for i in range(n, n - r, -1):
        numer *= i
    denom = 1
    for i in range(1, r + 1):
        denom *= i
    return numer / denom

def binomial(n, p):
    q = 1 - p
    y = [cons(n, k) * (p ** k) * (q ** (n - k)) for k in range(n)]
    return y


def separate_classes(class_: list) -> dict:
    sep = {}

    for i in range(len(class_)):
        v = class_[i]
        cls_val = v[-1]
        if cls_val not in sep:
            sep[cls_val] = []
        sep[cls_val].append(v)
    return sep


def summarize_data(classes: np.array) -> list:
    summaries = []
    col_len = len(classes)
    for col in classes.T:
        summaries.append((np.mean(col), np.std(col), col_len))

    return summaries


def summarize_class(classes: list) -> list:
    summaries = []

    for row in classes:
        summaries.append(summarize_data(row))
    return summaries


def norm_pdf(x: float, mu: float=0, sigma: float=1) -> float:
    #exp = np.exp(-(x-mu)**2 / (2 * sigma**2))
    sigma_squared = sigma ** 2
    if sigma <= 0:
        return 0.0
    pdf = (1 / np.sqrt(2 * np.pi * sigma_squared)) * np.e ** ((-0.5) * (x - mu) ** 2 / sigma_squared)
    return pdf
    #return (1 / (np.sqrt(2 * np.pi) * sigma)) * exp


def calc_class_probability(vocab: list, r: list):
    """
        P(Class | word, word, ....)
    :param data_summary:
    :param r:
    :return:
    """
    row_count = sum([vocab[word][0][2] for word in vocab])
    probs = 0

    for row in vocab:
        probs = vocab[r] / float(row_count)
        for i in range(len(row)):
            mean, stdev, _ = row[i]
            probs *= norm_pdf(r[i], mean, stdev)
    return probs


def print_err(*args, **kwargs):
    """Print to stderr
    :param args: arguments
    :param kwargs: separator
    """
    print(*args, file=sys.stderr, **kwargs)


def catch_err():
    exc_type, exc_obj, exc_tb = sys.exc_info()
    f_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print_err(exc_type, f_name, exc_tb.tb_lineno)
    exit(1)


def _check_partial_first_call(clf, classes=None):
    if getattr(clf, "classes", None) is None and classes is None:
        raise ValueError("Must pass class on first call")
    elif classes is not None:
        if getattr(clf, 'classes', None) is not None:
            if not np.array_equal(clf.classes, get_unique_labels(classes)):
                raise ValueError("Class not the same as previous")
            else:
                clf.classes = get_unique_labels(classes)
                return True

    return False


def get_unique_labels(*lbls):
    if not lbls:
        raise ValueError("No arguments passed in!")

    _unique_labels = []
    for i in lbls:
        _unique_labels = list(set(_unique_labels + i))

    return np.array(sorted(_unique_labels))
