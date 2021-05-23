import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
import sys

from classifier import *
#from classification import *
from pre_process import *

def print_test_results(results, test_docs):
    misclassified_docs = []
    for result, doc in zip(results, test_docs):
        if result != doc[1]:
            misclassified_docs.append(doc + [result])

    # for doc in misclassified_docs:
    #     print('misclassified {}: actual: {}, expected: {}'.format(doc[0], doc[2], doc[1]))

    correct = len(test_docs) - len(misclassified_docs)
    incorrect = len(misclassified_docs)
    accuracy = correct / len(test_docs)

    print("\n")
    print("****************************************")
    print("* RESULTS:")
    print("*     Correct:   {}".format(correct))
    print("*     Incorrect: {}".format(incorrect))
    print("*")
    print("*     Accuracy:  {}".format(accuracy))
    print("****************************************")


def test(test_docs, prior, likelihood, classes, vocab):
    results = []
    for test_doc in test_docs:
        results.append(test_naive_bayes(test_doc, prior, likelihood, classes, vocab))
    print_test_results(results, test_docs)

def test2(test_docs, prior, likelihood, classes, vocab, test_bayes):
    results = []
    for test_doc in test_docs:
        results.append(test_bayes(test_doc))
    print_test_results(results, test_docs)

def main():
    # pos_test, neg_test, test_docs = preprocess("testSet.txt", "preprocessed_test.txt")
    #pos_train, neg_train, train_docs, train_df = preprocess("trainingSet.txt", "preprocessed_train.txt")
    x_test, y_test, vocab_test, test_docs = preprocess("testSet.txt", "preprocessed_test.txt")
    #train_df = preprocess("trainingSet.txt", "preprocessed_train.txt")
    x_train, y_train, vocab, docs = preprocess("trainingSet.txt", "preprocessed_train.txt")
    #vocab = list(set(pos_train) | set(neg_train))

    bayes = Classifier(x_train, y_train, vocab, docs)

    prior, likelyhood = train_naive_bayes(vocab, docs, [0,1])
    bayes.train_naive_bayes()

    test2(test_docs, prior, likelyhood, [0, 1], vocab, bayes.test_naive_bayes)
    test(test_docs, prior, likelyhood, [0, 1], vocab)
    #print(prior,"\n", likelyhood)
    #print(likelyhood)
    #test(test_docs, prior, likelyhood, [0, 1], vocab)


if __name__ == "__main__":
    main()
