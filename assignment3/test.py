import doctest
from classifier import *

if __name__ == "__main__":
    df = pd.DataFrame
    doctest.testfile("classifier.py", extraglobs={"c": Classifier([], df, [], [])})
