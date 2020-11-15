
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

KN_MODEL = "kn_classifier"
ADA_MODEL = "adaboost_classifier"

CLASS_LABEL = 'outcome'


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


def load_model(name, generate_model):
    try:
        with open(name+'.pkl', 'rb') as fd:
            model = joblib.load(fd)
            return model
    except Exception as e:
        print("'{}' could not be found".format(name))
        return generate_model()


def store_model(name, model):
    try:
        with open(name+'.pkl', 'wb') as fd:
            joblib.dump(model, fd)
            return True
    except Exception as e:
        print(e)
        print("Unable to store model '{}'".format(name))
        return False


def apply_scheme(df):
    for k, v in SCHEMA.items():
        df[k] = df[k].astype(v)
    for i in TIME_FEATURES:
        df[i] = df[i].view('int64')
    return df.drop(columns=DROP_FEATURES)


def generate_KN_model():
    return KNeighborsClassifier()


def generate_boosted_tree():
    return AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=200)


def train_and_fit_models(X_test, X_train, y_test, y_train, model_meta):
    for name, generator in model_meta:
        print("\nTraining {} ...".format(name))
        model = generator()
        model.fit(X_train, y_train)
        print("Scores for {} ...".format(name))
        print("- Training accuracy: {}".format(model.score(X_train, y_train)))
        print("- Test accuracy: {}".format(model.score(X_test, y_test)))
        print("Storing model {}\n".format(name))
        joblib.dump(model, name+'.pkl')


def main(data_file, load_existing_models=False):
    df = pd.read_csv(data_file).sample(2000)  # TODO: Remove in order to train on entire dataset
    df = apply_scheme(df)

    encoded_df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES)
    features = list(encoded_df.columns)
    features.remove(CLASS_LABEL)

    imputer = SimpleImputer(strategy='mean')  # TODO: change this to be more meaningful imputation method
    encoded_df['age'] = imputer.fit_transform(encoded_df[['age']])

    scaler = MinMaxScaler()
    df[SCALED_FEATURES] = scaler.fit_transform(df[SCALED_FEATURES], df[CLASS_LABEL])

    X_train, X_test, y_train, y_test = train_test_split(encoded_df[features], encoded_df[CLASS_LABEL], test_size=0.2, random_state=12)

    model_meta = list(zip([KN_MODEL, ADA_MODEL], [generate_KN_model, generate_boosted_tree]))
    train_and_fit_models(X_test, X_train, y_test, y_train, model_meta)


if __name__ == '__main__':
    main(sys.argv[1])
