from matplotlib import pyplot as plt
from classifier import *
from pre_process import *
from sklearn.metrics import confusion_matrix


def main():
    x_test, y_test, vocab_test, test_docs = preprocess("testSet.txt", "preprocessed_test.txt")
    x_train, y_train, vocab, docs = preprocess("trainingSet.txt", "preprocessed_train.txt")

    bayes = Classifier(x_train, y_train, vocab, docs)
    bayes.train_naive_bayes()

    print("\nTesting training data:")
    res1 = test(docs, bayes)
    print("\nTesting test data:")
    res2 = test(test_docs, bayes)
    #to_chart(res1, res2)


def test(test_docs, bayes):
    y_pred = []
    y_true = []
    for test_doc in test_docs:
        y_true.append(test_doc[1])
        y_pred.append(bayes.test_naive_bayes(test_doc))

    #bayes.update_priors(y_true, y_pred)

    incorrect = 0
    for tt, ff in zip(y_true, y_pred):
        if tt != ff:
            incorrect += 1

    correct = len(test_docs) - incorrect
    incorrect = incorrect
    accuracy = correct / len(test_docs)
    cm = confusion_matrix(y_true, y_pred)

    print("-----------------------------------------")
    print("- Results:\n-\t\t# of correct:       %s\n-\t\t# of incorrect:     %s\n-\t\tAccuracy:\t\t\t%s%%"
          % (correct, incorrect, round(accuracy * 100, 2)))
    print("\n- Confusion matrix:\n")
    print("\tpred_neg   pred_pos\nneg\t\t%s\t\t%s\npos\t\t%s\t\t%s" % (cm[0][0], cm[0][1], cm[1][0], cm[1][1]))
    print("-----------------------------------------")
    return y_pred


def to_chart(result1, result2):
    plt.plot(result1, result2)
    plt.show()

if __name__ == "__main__":
    main()
