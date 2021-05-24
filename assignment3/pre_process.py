import sys, os
import re


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
        sent_vec = []
        with open(in_file, mode="r") as f:
            # Build set of`
            for line in f:
                line = line.rsplit(sep="\t")
                sent = line[1].strip(" \n")
                sent = int(sent)

                sentence = process_sentence(line[0])

                for w in sentence:
                    if w in stop_words or w == "":
                        sentence.remove(w)

                corpus.append([sentence, sent])
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
        corpus = sorted(corpus)

        vocab = sorted(vocab)
        vocab.append("classlabel")
        to_file(file_out, vocab, corpus)

        return x, y, vocab, corpus
    except:
        catch_err()


def process_word(word: str) -> str:
    return re.compile(r'[\W_]+').sub('', word.lower())


def process_sentence(sentence: str) -> list:
    return [process_word(word) for word in sentence.split()][:-1]


def to_file(file_name: str, vocab, corpus: list):
    with open(file_name, mode="w", encoding="UTF-8") as out_file:
        out_file.write(','.join(vocab) + '\n' +
                       '\n'.join([format_corpus(s, vocab) for s in corpus]))
    out_file.close()


def format_corpus(sentence: dict, vocab: dict) -> str:
    return (
        ','.join(['1' if word in sentence[0] else '0' for word in vocab])
        + ',' + str(sentence[1])
    )


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
