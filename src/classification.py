
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
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
        print("Storing model {} ...".format(name))
        pickle.dump(model, open(name+'.pkl', 'wb'))
        print("Scores for {} ...".format(name))
        print("- Training accuracy: {}".format(model.score(X_train, y_train)))
        print("- Test accuracy: {}\n".format(model.score(X_test, y_test)))


def load_and_fit_models(X_test, X_train, y_train, y_test, model_meta):
    for name, _ in model_meta:
        print("Loading {} ...".format(name))
        model = pickle.load(open(name + '.pkl', 'rb'))
        print("Scores for {} ...".format(name))
        print("- Training accuracy: {}".format(model.score(X_train, y_train)))
        print("- Test accuracy: {}\n".format(model.score(X_test, y_test)))

        y_predict = model.predict(X_test)
        print(confusion_matrix(y_test, y_predict))
        print(classification_report(y_test, y_predict))


def main(data_file, load_existing_models=False):
    df = pd.read_csv(data_file)
    df = apply_scheme(df)

    encoded_df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES)
    features = list(encoded_df.columns)
    features.remove(CLASS_LABEL)

    imputer = SimpleImputer(strategy='mean')  # TODO: change this to be more meaningful imputation method
    encoded_df['age'] = imputer.fit_transform(encoded_df[['age']])

    scaler = MinMaxScaler()
    df[SCALED_FEATURES] = scaler.fit_transform(df[SCALED_FEATURES], df[CLASS_LABEL])

    X_train, X_test, y_train, y_test = train_test_split(encoded_df[features], encoded_df[CLASS_LABEL], test_size=0.2)

    model_meta = list(zip([KN_MODEL, ADA_MODEL], [generate_KN_model, generate_boosted_tree]))
    if load_existing_models:
        load_and_fit_models(X_test, X_train, y_train, y_test, model_meta)
    else:
        train_and_fit_models(X_test, X_train, y_test, y_train, model_meta)


if __name__ == '__main__':
    load_models = False
    if sys.argv[-1] == '--load-existing-models':
        load_models = True
    main(sys.argv[1], load_existing_models=load_models)
