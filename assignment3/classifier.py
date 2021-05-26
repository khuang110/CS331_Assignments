import numpy as np
from collections import defaultdict
import sys, os


class Classifier:
    def __init__(self, vocab: list, corpus: list):
        """
        Bayes net classifier class
        :param vocab:
        :param corpus:
        """
        self._classes = [0, 1]
        self._vocab = vocab
        self._corpus = corpus
        self.sent_vocab = {
            1: self.vectorizer(self.flatten_words(1)),
            0: self.vectorizer(self.flatten_words(0)),
        }
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

    def count(self, cls_label=None) -> int:
        """
        Get count of class elements
        :param cls_label: class label <1, 0>
        :return: count in given class
        """
        if cls_label is None:
            if hasattr(self, "_total_count"):
                if cls_label is None:
                    return getattr(self, "_total_count")
            setattr(self, "_total_count", len(self._corpus))
            return getattr(self, "_total_count")
        elif cls_label == 1 or cls_label == 0:
            if hasattr(self, "_count" + str(cls_label)):
                return getattr(self, "_count" + str(cls_label))
            setattr(
                self,
                "_count" + str(cls_label),
                len([s for s in self._corpus if s[1] == cls_label]),
            )
            return getattr(self, "_count" + str(cls_label))
        else:
            raise ClassifierException("Invalid class label!!")

    def word_count(self, cls_label: int) -> int:
        """
        Returns count of words in a given class
        :param cls_label: <1, 0>
        :return: number of words in given class : int
        """
        if cls_label == 0 or cls_label == 1:
            if hasattr(self, "_word_count" + str(cls_label)):
                return getattr(self, "_word_count" + str(cls_label))
            setattr(
                self,
                "_word_count" + str(cls_label),
                sum(self.flatten_words(cls_label).count(w) for w in self._vocab),
            )
            return getattr(self, "_word_count" + str(cls_label))

    def flatten_words(self, cls_label: int) -> list:
        """
        Turns list of list containing sentences into list of words
        :param cls_label: class to get words from <1, 0>
        :return: list of words : list

        Examples:
        corpus: [["i hi foo", 0], ["hi foo foo", 1], ["foo", 0]]
        cls_label = 0
        ["i", "hi", "foo", "foo"]
        """
        return list(np.concatenate([s[0] for s in self._corpus if s[1] == cls_label]))

    def tokenize_corpus(self) -> dict:
        """
        Takes corpus (list of list) and turns into dict with count of words
        :return: word counts in corpus : dict

        Examples:
        corpus: [["i hi foo", 0], ["hi foo foo", 1]]
        {"i": 1, "hi": 2, "foo": 3}
        """
        return self.vectorizer(self.flatten_words(0) + self.flatten_words(1))

    def train_naive_bayes(self):
        """
        Train the bayesian network
        :return: sets class variables used to test
        """
        for cls_label in self._classes:
            normal_prior = np.log(self.count(cls_label) / self.count())

            for word in self._vocab:
                self.prior_probability[cls_label] = normal_prior
                word_count = self.sent_vocab[cls_label][word]
                self.conditional_probability[word][cls_label] = np.log(
                    (word_count + 1) / (self.word_count(cls_label) + 1)
                )

    def test_naive_bayes(self, sentence: list) -> int:
        """
        Test the naive bayes net
        :param sentence: Sentence to test
        :return: Class prediction: <1, 0>
        """
        if not self.prior_probability or not self.conditional_probability:
            raise ClassifierException("Must train bayes net before testing")

        sentence = [word for word in sentence[0] if word in self._vocab]

        prediction = {0: 0, 1: 0}

        for cls_label in self._classes:
            if not sentence:
                continue
            prediction[cls_label] = self.prior_probability[cls_label]

            for word in sentence:
                prediction[cls_label] += self.conditional_probability[word][cls_label]

        return 0 if prediction[0] > prediction[1] else 1

    @staticmethod
    def to_string(correct, incorrect, accuracy, cm=None):
        """
        Print results to string
        :param correct: # of correct predictions
        :param incorrect: # of incorrect predictions
        :param accuracy: % accurate
        :param cm: Confusion matrix

        """
        print("-----------------------------------------")
        print(
            "- Results:\n-\t\t# of correct:       %s\n-\t\t# of incorrect:     %s\n-\t\tAccuracy:\t\t%s%%"
            % (correct, incorrect, round(accuracy * 100, 2))
        )
        if cm is not None:
            print("\n- Confusion matrix:\n")
            print(
                "\tpred_neg   pred_pos\nneg\t\t%s\t\t%s\npos\t\t%s\t\t%s"
                % (cm[0][0], cm[0][1], cm[1][0], cm[1][1])
            )
        print("-----------------------------------------")

    @classmethod
    def bayes_to_file(cls, **kwargs):
        """
        Redirect print results to file
        :param kwargs: to_string arguments
        :return: results.txt : file
        """
        # Save stdout
        stdout = sys.stdout
        f_name = "results.txt"
        f_size = 0
        if os.path.isfile(f_name):
            f_size = os.path.getsize(f_name)
        if f_size <= 175:
            mode = "a+"
        else:
            mode = "w+"
        # Write stdout to file
        with open(f_name, mode=mode) as out:
            sys.stdout = out
            cls.to_string(**kwargs)
            sys.stdout = stdout


class ClassifierException(Exception):
    """
    Error messages for Classifier class
    """
    data = {}
    message = "An unknown error occured"

    def __init__(self, message=None, data={}):
        if message:
            self.message = message
        if data:
            self.data = data

    def __str__(self):
        if self.data:
            return ":: {}\n::{}".format(self.data, self.message)
        return ":: {}".format(self.message)
