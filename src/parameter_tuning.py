
import os
import sys
import time
import warnings
import itertools
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split


warnings.filterwarnings('ignore')  # sklearn: Precision and F-score are ill-defined and being set to 0.0 warning

RANDOM_SEED = 31415  # first 5 digits of pi
CLASS_LABEL = 'outcome'
LOG_DIR = './log'

SCHEMA = {'age': 'float64', 'sex': 'category', 'province': 'category',
          'country': 'category', 'latitude': 'float64', 'longitude': 'float64',
          'date_confirmation': 'datetime64[ns]',
          'outcome': 'category', 'epoch_date_confirmation': 'datetime64[ns]',
          'Province_State': 'category', 'Country_Region': 'category', 'Last_Update': 'datetime64[ns]',
          'Lat': 'float64', 'Long_': 'float64', 'Confirmed': 'int64', 'Deaths': 'int64', 'Recovered': 'int64',
          'Active': 'int64', 'Incidence_Rate': 'float64', 'Case-Fatality_Ratio': 'float64'}

CATEGORICAL_FEATURES = ['sex', 'province', 'country', 'Province_State', 'Country_Region']
TIME_FEATURES = ['date_confirmation', 'epoch_date_confirmation', 'Last_Update']
DROP_FEATURES = ['Last_Update', 'additional_information', 'source', 'Combined_Key']
SCALED_FEATURES = ['age', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Incidence_Rate', 'Case-Fatality_Ratio']


KNN_GRID = {'leaf_size': np.unique(np.geomspace(5, 50, num=3).astype(int)),
            'n_neighbors': np.unique(np.linspace(5, 9, num=3).astype(int))
            }

ADA_GRID = {'base_estimator__max_depth': np.unique(np.geomspace(2, 40, num=5)).astype(int),
            'learning_rate': np.unique(np.linspace(0.5, 2, num=3))
            }


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(" **Time: {} took {}".format(method.__name__, str(timedelta(seconds=te-ts))))
        return result
    return timed


@timeit
def apply_scheme(df):
    for k, v in SCHEMA.items():
        df[k] = df[k].astype(v)
    for i in TIME_FEATURES:
        df[i] = df[i].view('int64')
    return df.drop(columns=DROP_FEATURES)


@timeit
def grid_search(X_test, X_train, y_test, y_train, model, grid, verbose=False, is_boosted=False):
    first_param, second_param = grid.values()
    first_key, second_key = grid.keys()
    combinations = itertools.product(first_param, second_param)
    scoring = {'accuracy': make_scorer(accuracy_score),
               'weighted_recall': make_scorer(recall_score, average='weighted'),
               'recall': make_scorer(recall_score, average='macro'),
               'deceased_recall': make_scorer(recall_score, average=None, labels=['deceased'])}

    manifest = {first_key: [], second_key: [], 'cross validation score': [], 'macro recall': [], 'deceased recall': [], 'micro recall': []}
    for i, j in combinations:
        print(f" -Trying: {first_key}={i}, {second_key}={j}")
        if is_boosted:
            m = model(DecisionTreeClassifier(), **{first_key: i, second_key: j})
        else:
            m = model(**{first_key: i, second_key: j})
        scores = cross_validate(m, X_train, y_train, cv=3, scoring=scoring)

        if verbose:
            print(f'  --Cross validated mean accuracy: {scores["test_accuracy"].mean()}')

            print(f"  --Training 'deceased' recall={scores['test_deceased_recall'].mean()} |"
                  f" macro recall {scores['test_recall'].mean()} |"
                  f" micro recall {scores['test_weighted_recall'].mean()}")

        manifest[first_key].append(i)
        manifest[second_key].append(j)
        manifest['cross validation score'].append(scores["test_accuracy"].mean())
        manifest['macro recall'].append(scores['test_recall'].mean())
        manifest['deceased recall'].append(scores['test_deceased_recall'].mean())
        manifest['micro recall'].append(scores['test_weighted_recall'].mean())
        del m
    return manifest


def _get_best_score(manifest, metric):
    max_index = manifest[metric].index(max(manifest[metric]))
    return [manifest[i][max_index] for i in manifest.keys()]


def get_best_params(manifest):
    metrics = ['cross validation score', 'macro recall', 'deceased recall', 'micro recall']
    keys = manifest.keys()
    scores = [_get_best_score(manifest, i) for i in metrics]

    print(' | '.join(keys))
    for i, score in zip(metrics, scores):
        print('Best {}: {}'.format(i, score))

@timeit
def grid_search_adaboost(X_test, X_train, y_test, y_train, grid):
    ada = AdaBoostClassifier(DecisionTreeClassifier())
    clf = GridSearchCV(ada, grid, cv=2, verbose=100000, n_jobs=2)
    clf.fit(X_train, y_train)
    return clf

@timeit
def impute_by_mean(df):
    dfs = dict(tuple(df.groupby('outcome')))
    recovered = dfs['recovered']
    deceased = dfs['deceased']
    nonhospitalized = dfs['nonhospitalized']
    hospitalized = dfs['hospitalized']

    imputer = SimpleImputer(strategy='mean')
    recovered['age'] = imputer.fit_transform(recovered[['age']])
    deceased['age'] = imputer.fit_transform(deceased[['age']])
    hospitalized['age'] = imputer.fit_transform(hospitalized[['age']])
    nonhospitalized['age'] = imputer.fit_transform(nonhospitalized[['age']])

    df_list = [recovered, deceased, hospitalized, nonhospitalized]
    concat_df = pd.concat(df_list)
    return concat_df


def main(data_file):
    start = time.time()
    print("== Starting Execution ==")
    df = pd.read_csv(data_file)#.sample(1000, random_state=RANDOM_SEED)  # TODO: remove

    print("Encoding data ...")
    df = apply_scheme(df)
    encoded_df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES)
    features = list(encoded_df.columns)
    features.remove(CLASS_LABEL)

    encoded_df = impute_by_mean(encoded_df)

    scaler = MinMaxScaler()
    df[SCALED_FEATURES] = scaler.fit_transform(df[SCALED_FEATURES], df[CLASS_LABEL])

    X_train, X_test, y_train, y_test = train_test_split(encoded_df[features], encoded_df[CLASS_LABEL],
                                                        test_size=0.25, random_state=RANDOM_SEED)

    # print('Running KNN Grid Search ...')
    # manifest = grid_search(X_test, X_train, y_test, y_train, KNeighborsClassifier, KNN_GRID, verbose=True)

    print('Running ADA Grid Search ...')
    manifest = grid_search(X_test, X_train, y_test, y_train, AdaBoostClassifier, ADA_GRID, verbose=True, is_boosted=True)

    print("== Total running time {} ==\n".format(str(timedelta(seconds=time.time()-start))))

    return manifest


if __name__ == '__main__':
    manifest = main(sys.argv[1])
    results = pd.DataFrame(manifest)
    results.to_csv(os.path.join(f"results_{time.time()}.csv"))
    get_best_params(manifest)
