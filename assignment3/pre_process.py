import sys, os
import re
import pickle


def preprocess(in_file: str, file_out: str) -> tuple:
    """
    Preprocess data, clean out separators, sort in alphabetical order
    :param in_file: Input file name
    :param file_out: Output file name
    :return: Sorted cleaned data
    """
    try:
        stop_words = []
        if os.path.isfile("StopWords.pkl"):
            with open("StopWords.pkl", mode="rb") as pkl:
                stop_words = pickle.load(pkl)
        else:
            with open("StopWords.txt", mode="r") as f1:
                for line in f1:
                    stop_words.append(line.strip("\n"))
            with open("StopWords.pkl", mode="wb") as pkl:
                pickle.dump(stop_words, pkl)

        vocab = []
        corpus = []
        with open(in_file, mode="r") as f:
            for line in f:
                line = line.rsplit(sep="\t")
                sent = line[1].strip(" \n")
                sent = int(sent)

                sentence = process_sentence(line[0])

                [sentence.remove(w) for w in sentence if w in stop_words or w == ""]
                corpus.append([sentence, sent])
                vocab += list(set(sentence) - set(vocab))

        # Sort lists alphabetically
        corpus = sorted(corpus)
        vocab = sorted(vocab)
        vocab.append("classlabel")
        to_file(file_out, vocab, corpus)

        return vocab, corpus
    except:
        catch_err()


def process_word(word: str) -> str:
    """
    Remove special characters in word
    :param word: a word
    :return: word only alphabet, number

    Examples:

    >>> process_word("f&&^oo%#$#bar")
    'foobar'

    >>> process_word("#ff$f")
    'fff'
    """
    return re.compile(r"[\W_]+").sub("", word.lower())


def process_sentence(sentence: str) -> list:
    """
    Turn sentence into list of words and remove special char,
    delimits with space. Words that contain numbers will have numbers removed.
    Numbers are only left if they are not attached to word.
    :param sentence: sentence to process
    :return: list of words : lsit

    Examples:
    >>> process_sentence("hi the^e f$$ 100 !")
    ['hi', 'thee', 'f', '100']

    >>> process_sentence("foo bar !@#$#")
    ['foo', 'bar']
    """
    return [process_word(word) for word in sentence.split()][:-1]


def to_file(file_name: str, vocab, corpus: list):
    """
    Output pre processed data to file. data will be processed without stop-words
    and no special chars
    :param file_name: name of output file
    :param vocab: total vocab no repeats
    :param corpus: all of the sentences
    :return: file
    """
    with open(file_name, mode="w", encoding="UTF-8") as out_file:
        out_file.write(
            "".join(vocab[0])
            + ",".join(vocab[1:])
            + "\n"
            + "\n".join([format_corpus(s, vocab) for s in corpus])
        )


def format_corpus(sentence: list, vocab: list) -> str:
    """
    Formats the sentence to be delimited by comma for output file
    :param sentence: sentence to format
    :param vocab: list of all words
    :return: 1 or 0 in place of word if it is present in sentence

    Examples:
    >>> format_corpus(["my name is foo", 1], ["beta", "cat", "foo", "is", "my", "name", "zoo"])
    '0,0,1,1,1,1,0,1'
    """
    return (
        ",".join(["1" if word in sentence[0] else "0" for word in vocab])
        + ","
        + str(sentence[1])
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


if __name__ == "__main__":
    import doctest
    doctest.testmod()
