

import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Process
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score, accuracy_score, precision_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

KN_MODEL = "kn_classifier"
ADA_MODEL = "adaboost_classifier"
TITLE = {KN_MODEL: "K-Nearest Neighbor Classifier", ADA_MODEL: "AdaBoost Decision Tree Classifier"}

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


def load_and_fit_models(X_test, X_train, y_train, y_test, model_meta):
    for name, m in model_meta:
        print(name)
        print(m)
        print("Loading {} ...".format(name))
        model = m.fit(X_train, y_train)
        print("Scores for {} ...".format(name))
        print("- Training accuracy: {}".format(model.score(X_train, y_train)))
        print("- Test accuracy: {}\n".format(model.score(X_test, y_test)))

        print("Generating Confusion Matrix ...")
        y_predict = model.predict(X_test)
        y_predict_train = model.predict(X_train)
        plot_confusion_matrix(model, X_test, y_test)
        plt.title("{} Confusion Matrix".format(TITLE[name]))
        plt.show()
        print("Creating classification report")
        print("- Training report")
        print(classification_report(y_train, y_predict_train))
        print("- Testing report")
        print(classification_report(y_test, y_predict))


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


def main(data_file, load_existing_models=False):
    df = pd.read_csv(data_file)
    df = apply_scheme(df)

    encoded_df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES)
    features = list(encoded_df.columns)
    features.remove(CLASS_LABEL)

    encoded_df = impute_by_mean(encoded_df)

    scaler = MinMaxScaler()
    df[SCALED_FEATURES] = scaler.fit_transform(df[SCALED_FEATURES], df[CLASS_LABEL])

    X_train, X_test, y_train, y_test = train_test_split(encoded_df[features], encoded_df[CLASS_LABEL], test_size=0.2)

    model_meta = list(zip([KN_MODEL, ADA_MODEL], [KNeighborsClassifier(), AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=200)]))
    load_and_fit_models(X_test, X_train, y_train, y_test, model_meta)


if __name__ == '__main__':
    load_models = False
    main(sys.argv[1], load_existing_models=load_models)
