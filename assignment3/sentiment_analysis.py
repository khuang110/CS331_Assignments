import pandas as pd
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import sys

from pre_process import *


def main():
    preprocess('testSet.txt', 'preprocessed_test.txt')
    preprocess('trainingSet.txt', 'preprocessed_train.txt')


if __name__ == "__main__":
    main()
