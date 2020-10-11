
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FIGURE_DIR = '../figures'


def save_barh_figure(x, y, title, xlabel, ylabel, img_title):
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches((8.5, 11), forward=False)
    plt.barh(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(img_title, dpi=500, bbox_inches='tight')


def deaths_by_country(location_df):
    # TODO: confirmed cases, recovery by country
    country_df = location_df[['Country_Region', 'Deaths', 'Confirmed', 'Recovered']].groupby('Country_Region').sum()

    deaths = country_df[country_df['Deaths'] > 1000].sort_values('Deaths')  # Graph gets messy without this restriction
    confirmed = country_df[country_df['Confirmed'] > 50000].sort_values('Confirmed')
    recovered = country_df[country_df['Recovered'] > 50000].sort_values('Recovered')

    save_barh_figure(deaths.index, deaths['Deaths'], 'Deaths by Country', 'Number of Deaths',
                     'Country', os.path.join(FIGURE_DIR, 'deaths_by_country.png'))
    save_barh_figure(confirmed.index, confirmed['Confirmed'], 'Confirmed Cases by Country',
                     'Number of Cases', 'Country',   os.path.join(FIGURE_DIR, 'cases_by_country.png'))
    save_barh_figure(recovered.index, recovered['Recovered'], 'Recoveries by Country',
                     'Number of Recoveries', 'Country',   os.path.join(FIGURE_DIR, 'recoveries_by_country.png'))


def cases_by_age():
    # TODO: deaths, confirmed cases, recovery by age group
    pass


def compute_missing_values(df):
    missing = []
    for col in df.columns:
        m = df[col].isna().sum()
        missing.append([col, m])
    missing_df = pd.DataFrame(missing, columns=['attribute', 'missing_count'])
    return missing_df


def main(individual_file, location_file):
    individual_df = pd.read_csv(individual_file)
    location_df = pd.read_csv(location_file, parse_dates=['Last_Update'])

    print(individual_df.describe())
    print(location_df.describe())

    print(compute_missing_values(individual_df))
    print(compute_missing_values(location_df))
    deaths_by_country(location_df)


if __name__ == '__main__':
    assert len(sys.argv) == 3
    individual_file = sys.argv[1]
    location_file = sys.argv[2]
    main(individual_file, location_file)