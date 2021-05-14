import json


def preprocess(in_file, file_out):
    """
    Preprocess data, clean out separators, sort in alphabetical order
    :param in_file: Input file name
    :param file_out: Output file name
    :return: Sorted cleaned data
    """
    try:
        sentiment = {0: [], 1: []}
        with open(in_file, mode='r') as f:

            # Build set of`
            for line in f:
                line = line.rsplit(sep='\t')
                sent = line[1].strip(' \n')
                sent = int(sent)
                # Validate second part of line is int
                if sent == 0 or sent == 1:
                    data = []
                    word = ""
                    for i in line[0]:
                        if i == ' ':
                            if word not in data and word != '':
                                data.append(word.lower())
                            word = ""
                        elif i.isalnum():
                            word += i

                    # Concat list and remove dups
                    sentiment[sent] = sentiment[sent] + list(set(data) - set(sentiment[sent]))
            f.close()

            # Sort lists alphabetically
            sentiment[0] = sorted(sentiment[0])
            sentiment[1] = sorted(sentiment[1])
            return sentiment
    except:
        print_err(sys.exc_info()[0], "occurred!")
        exit(1)


def to_file(file_name, data):
    with open(file_name, mode='w') as out_file:
        json.dump(data, out_file)
    out_file.close()


def print_err(*args, **kwargs):
    """ Print to stderr
    :param args: arguments
    :param kwargs: separator
    """
    print(*args, file=sys.stderr, **kwargs)