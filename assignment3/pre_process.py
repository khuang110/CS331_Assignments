import json
import sys, os
import pandas as pd
import numpy as np
from collections import defaultdict
import re


def extract_word_features(s: str) -> defaultdict:
    """
    Extract features from given sentence.
        word is delimited by space character
    :param s: input word
    :return: dict with word counts

    >>> extract_word_features("I see I what")
    {"i": 2, "see": 1, "what": 1}
    """

    word_dict = defaultdict(float)
    for word in s.split():
        word_dict[word] += 1
    return word_dict


def process_word(word: str) -> str:
    return re.compile(r'[\W_]+').sub('', word.lower())


def process_sentence(sentence: str) -> list:
    return [process_word(word) for word in sentence.split()][:-1]


def preprocess(in_file: str, file_out: str) -> tuple:
    """
    Preprocess data, clean out separators, sort in alphabetical order
    :param in_file: Input file name
    :param file_out: Output file name
    :return: Sorted cleaned data
    """
    try:
        stop_words = []
        with open("StopWords.txt", mode="r") as f1:
            for line in f1:
                stop_words.append(line.strip("\n"))

        x = []
        y = []
        vocab = []
        corpus = []
        corpus2 = []
        sent_vec = []
        with open(in_file, mode="r") as f:
            # Build set of`
            for line in f:
                line = line.rsplit(sep="\t")
                sent = line[1].strip(" \n")
                sent = int(sent)

                sentence = process_sentence(line[0])
                corpus.append(sentence)
                corpus2.append([sentence, sent])

                for w in sentence:
                    if w in stop_words or w == "":
                        sentence.remove(w)

                # Concat list
                if sent == 1:
                    x += list(set(sentence) - set(x))
                else:
                    y += list(set(sentence) - set(y))
                sent_vec.append(sent)

                vocab += list(set(sentence) - set(vocab))

            f.close()

        # Sort lists alphabetically
        x = sorted(x)
        y = sorted(y)

        vocab = sorted(vocab)
        vocab.append("classlabel")
        feat_vecs = sentences_to_features(corpus, sent_vec, vocab)
        #df_features = pd.DataFrame([feat_vecs], columns=feat_vecs.keys())
        df_features = pd.concat({k: pd.Series(v) for k, v in feat_vecs.items()}, axis=1)
        df_doc = pd.DataFrame({"sentiment": sent_vec, "corpus": corpus}, columns=["sentiment", "corpus"])

        to_file(file_out, df_features)

        #return x, y, df_doc, df_features

        return x, y, vocab, sorted(corpus2)
    except:
        catch_err()


def sentences_to_features(sentences: list, sent_vec: list, vocab: list) -> dict:
    df = defaultdict(list)
    for sentence in sentences:
        x = vectorizer(sentence, vocab)
        for key, val in x.items():
            df[key].append(val)
    #df["classlabel"].append(sent_vec)

    return df


def vectorizer(sentence: list, vocab: list) -> dict:
    """Feature vectorizor

    :param sentence: array of classes (words).
        list of all the classes that can appear in a vector.
    :param vocab: List of all words
    :return: x : counts of words : defautdict

    Examples
    --------

    """
    x = defaultdict(int)
    for f in vocab:
        if f in sentence:
            x[f] = 1
        else:
            x[f] = 0
    return x


def get_stop_words():
    df = pd.read_csv("StopWords.txt")
    df.columns = ["StopWords"]
    df.to_pickle("StopWords.pkl")


def to_file(file_name, df):
    # with open(file_name, mode="w", encoding="UTF-8") as out_file:
    #     for key, data in kwargs.items():
    #         out_file.write("%s: %s\n" % (key, data))
    #out_file.close()
    df.to_csv(file_name, index=False)


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


def preprocess2(in_file: str, file_out: str) -> tuple:
    """
    Preprocess data, clean out separators, sort in alphabetical order
    :param in_file: Input file name
    :param file_out: Output file name
    :return: Sorted cleaned data
    """
    try:
        stop_words = []
        with open("StopWords.txt", mode="r") as f1:
            for line in f1:
                stop_words.append(line.strip("\n"))

        x = []
        y = []
        vocab = []
        corpus = []
        df = pd.read_pickle("StopWords.pkl")
        sent_vec = []
        with open(in_file, mode="r") as f:
            # Build set of`
            for line in f:
                line = line.rsplit(sep="\t")
                sent = line[1].strip(" \n")
                sent = int(sent)
                sentence = ''.join(re.findall('[a-zA-Z\s]+', line[0]))
                corpus.append(sentence.lower())
                # Validate second part of line is int
                if sent == 0 or sent == 1:
                    data = []
                    #word = ""
                    sent_vec.append(sent)
                    for word in line[0].rsplit(sep=" "):
                        #[^A-Za-z0-9]+
                        w = (''.join(re.findall('[a-zA-Z]+',word))).lower()
                        if w != "":
                            data.append(w)
                    # for i in line[0]:
                    #     if i == " ":
                    #         if word not in data and word != "": # and not df["StopWords"].str.contains(word).any():
                    #             data.append(word.lower())
                    #
                    #         word = ""
                    #     elif i.isalnum():
                    #         word.join(i)
                    # Vocab without duplicates
                    vocab += list(set(data) - set(vocab))
                    # Concat list
                    if sent == 1:
                        x += data
                    else:
                        y += data
            f.close()

        df_doc = pd.DataFrame({"sentiment": sent_vec, "corpus": corpus},columns=["sentiment", "corpus"])

        # Sort lists alphabetically
        x = sorted(x)
        y = sorted(y)
        vocab = sorted(vocab)
        vocab.append("classlabel")
        feat_vecs = sentences_to_features(corpus, sent_vec, vocab)
        df_features = pd.DataFrame([feat_vecs], columns=feat_vecs.keys())
        pos = []
        neg = []
        for i in range(len(x)):
            pos.append({"1": x[i]})
        for i in range(len(y)):
            neg.append({"0": y[i]})

        #to_file(file_out, corpus=corpus, positive_class=x, negative_class=y)
        df_sent = pd.DataFrame(list(zip(x, y)))
        return x, y, df_doc
    except:
        catch_err()