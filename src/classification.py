
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

MODEL_NAME = "m1"  # TODO: rename to be meaningful
MODEL_NAME2 = "m2"

CLASS_LABEL = 'outcome'


def load_model(name):
    try:
        with open(name, 'rb') as fd:
            model = pickle.load(fd)
            return model
    except Exception as e:
        print("'{}' could not be found".format(name))
        return None


def store_model(name, model):
    try:
        with open(name, 'rb') as fd:
            pickle.dump(model, fd, pickle.HIGHEST_PROTOCOL)
            return True
    except Exception as e:
        print("Unable to store model '{}'".format(name))
        return False


def main(data_file):
    df = pd.read_csv(data_file)
    features = list(df.columns)
    features.remove(CLASS_LABEL)

    X_train, X_test, y_train, y_test = train_test_split(df[features], df[CLASS_LABEL], test_size=0.2)


if __name__ == '__main__':
    main(sys.argv[1])
