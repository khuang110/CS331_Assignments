from classifier import *
from pre_process import *
#from sklearn.metrics import confusion_matrix


def main():
    vocab_test, test_docs = preprocess("testSet.txt", "preprocessed_test.txt")
    vocab, docs = preprocess("trainingSet.txt", "preprocessed_train.txt")

    print("Training naive bayes net...")
    bayes = Classifier(vocab, docs)
    bayes.train_naive_bayes()

    print("\nTesting training data:")
    test(docs, bayes)
    print("\nTesting test data:")
    test(test_docs, bayes)


def test(test_docs: list, bayes: Classifier):
    """
    Test the bayes net and get the results
    :param test_docs: sentences to test
    :param bayes: Pre trained bayes net : Classifier
    """
    y_pred = []
    y_true = []
    for test_doc in test_docs:
        y_true.append(test_doc[1])
        y_pred.append(bayes.test_naive_bayes(test_doc))

    incorrect = 0
    for tt, ff in zip(y_true, y_pred):
        if tt != ff:
            incorrect += 1

    correct = len(test_docs) - incorrect
    incorrect = incorrect
    accuracy = correct / len(test_docs)
    #cm = confusion_matrix(y_true, y_pred)
    bayes.to_string(correct, incorrect, accuracy)
    bayes.bayes_to_file(correct=correct, incorrect=incorrect, accuracy=accuracy)


if __name__ == "__main__":
    main()
