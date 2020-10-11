
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def deaths_by_country(location_df):
    country_df = location_df[['Country_Region', 'Deaths']].groupby('Country_Region').sum().sort_values('Deaths')
    country_df = country_df[country_df['Deaths'] > 1000]  # Graph gets messy without this restriction
    plt.barh(country_df.index, country_df['Deaths'])
    plt.title("Deaths by Country")
    plt.xlabel("Number of Deaths")
    plt.ylabel("Country")
    plt.show()


def compute_missing_values(df):
    missing = []
    for col in df.columns:
        m = df[col].isna().sum()
        missing.append([col, m])
    missing_df = pd.DataFrame(missing, columns=['attribute', 'missing_count'])
    return missing_df


def main(individual_file, location_file):
    individual_df = pd.read_csv(individual_file)
    location_df = pd.read_csv(location_file)

    print(compute_missing_values(individual_df))
    print(compute_missing_values(location_df))


if __name__ == '__main__':
    assert len(sys.argv) == 3
    individual_file = sys.argv[1]
    location_file = sys.argv[2]
    main(individual_file, location_file)