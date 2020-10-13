
import os
import re
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

FIGURE_DIR = '../figures'
RE_RANGE = re.compile(r'([0-9]+) ?- ?([0-9]+)')
RE_MONTH = re.compile(r'([0-9]+) month')
RE_DATE = re.compile(r'([0-9\.]+) ?- ?([0-9\.]+)')


def save_barh_figure(x, y, title, xlabel, ylabel, img_title):
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches((8.5, 11), forward=False)
    plt.barh(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(img_title, dpi=500, bbox_inches='tight')


def visual_by_country(location_df):
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


def visual_by_outcome(individual_df):
    sns.set_theme(style='whitegrid')
    dropped = individual_df.dropna()
    data = dropped.groupby(['outcome']).size().reset_index(name="count").sort_values('count', ascending=False)
    sns.barplot(x='outcome', y='count', data=data, palette='rocket')
    plt.title('Outcome Frequency')
    plt.xlabel('Outcome')
    plt.ylabel('Number of People')
    plt.savefig(os.path.join(FIGURE_DIR, 'outcomes.png'), dpi=500, bbox_inches='tight')


def compute_missing_values(df):
    missing = []
    for col in df.columns:
        m = df[col].isna().sum()
        missing.append([col, m])
    missing_df = pd.DataFrame(missing, columns=['attribute', 'missing_count'])
    return missing_df


def reduce_age_range(x):
    if pd.notnull(x):
        ranges = re.match(RE_RANGE, x)
        months = re.match(RE_MONTH, x)
        if ranges:
            lower, upper = ranges.groups()
            return (float(lower) + float(upper)) / 2  # ex case: 15 - 20
        elif months:
            return float(months.groups()[0]) / 12  # ex case: 13 months
        elif x.endswith('+'):
            return float(x[:2])  # ex case: 80+
        elif x.endswith('-'):
            return float(x[:2])  # ex case: 30 -
        else:
            return float(x)  # ex case: 30


def reduce_date_confirmation(x):
    if pd.notnull(x):
        ranges = re.match(RE_DATE, x)
        if ranges:
            lower, upper = ranges.groups()
            date_format = '%d.%m.%Y'
            initial_date = datetime.strptime(lower, date_format)
            final_date = datetime.strptime(upper, date_format)
            mean = (final_date - initial_date) / 2
            initial_date += mean
            return initial_date.strftime(date_format)
        return x


def detect_outliers(individual_df, location_df):
    lats_df = individual_df['latitude']
    # Get quartiles
    lats_quartiles = np.nanpercentile(lats_df, [25, 75])
    lats_first_quartile = lats_quartiles[0]
    lats_third_quartile = lats_quartiles[1]
    # Calc inter quartile range
    lats_inter_quartile_range = lats_third_quartile - lats_first_quartile
    # Calc outlier bounds
    lats_lower_bound = lats_first_quartile - lats_inter_quartile_range * 1.5
    lats_upper_bound = lats_third_quartile + lats_inter_quartile_range * 1.5
    # Filter df to get outliers
    outlier_lats_df = lats_df[(lats_df > lats_upper_bound) | (lats_df < lats_lower_bound)]
    # Get number of outliers
    outlier_lats_count = outlier_lats_df.count()

    print("outlier_lats_count = ", outlier_lats_count)


    longs_df = individual_df['longitude']
    # Get quartiles
    longs_quartiles = np.nanpercentile(longs_df, [25, 75])
    longs_first_quartile = longs_quartiles[0]
    longs_third_quartile = longs_quartiles[1]
    # Calc inter quartile range
    longs_inter_quartile_range = longs_third_quartile - longs_first_quartile
    # Calc outlier bounds
    longs_lower_bound = longs_first_quartile - longs_inter_quartile_range * 1.5
    longs_upper_bound = longs_third_quartile + longs_inter_quartile_range * 1.5
    # Filter df to get outliers
    outlier_longs_df = longs_df[(longs_df > longs_upper_bound) | (longs_df < longs_lower_bound)]
    # Get number of outliers
    outlier_longs_count = outlier_longs_df.count()

    print("outlier_longs_count = ", outlier_longs_count)


    date_time_df = pd.to_datetime(individual_df['date_confirmation'])
    date_time_no_outliers_df = remove_outliers(date_time_df)
    confirmed_no_outliers_df = remove_outliers(location_df['Confirmed'])
    deaths_no_outliers_df = remove_outliers(location_df['Deaths'])
    recovered_no_outliers_df = remove_outliers(location_df['Recovered'])
    active_no_outliers_df = remove_outliers(location_df['Active'])
    incidence_rate_no_outliers_df = remove_outliers(location_df['Incidence_Rate'])
    case_fatality_ratio_no_outliers_df = remove_outliers(location_df['Case-Fatality_Ratio'])

    print(date_time_no_outliers_df)
    print(confirmed_no_outliers_df)
    print(deaths_no_outliers_df)
    print(recovered_no_outliers_df)
    print(active_no_outliers_df)
    print(incidence_rate_no_outliers_df)
    print(case_fatality_ratio_no_outliers_df)

def remove_outliers(df):
    quantiles = df.quantile([.05, .95])
    lower_bound = quantiles[.05]
    upper_bound = quantiles[.95]
    print(quantiles)
    no_outliers_df = df[(df < upper_bound) | (df > lower_bound)]
    return no_outliers_df


def main(individual_file, location_file):
    individual_df = pd.read_csv(individual_file)
    individual_df['age'] = individual_df['age'].apply(reduce_age_range)
    individual_df['date_confirmation'] = individual_df['date_confirmation'].apply(reduce_date_confirmation)
    individual_df['date_confirmation'] = pd.to_datetime(individual_df['date_confirmation'])
    location_df = pd.read_csv(location_file, parse_dates=['Last_Update'])

    #print(individual_df.describe())
    #print(location_df.describe())

    #print(compute_missing_values(individual_df))
    #print(compute_missing_values(location_df))
    #visual_by_outcome(individual_df)
    #visual_by_country(location_df)

    # figure out compute func later
    compute_missing_values(individual_df)
    compute_missing_values(location_df)
    outliers = detect_outliers(individual_df, location_df)

def main1(individual_file, location_file):
    individual_df = pd.read_csv(individual_file)
    outliers = detect_outliers(individual_df)

if __name__ == '__main__':
    assert len(sys.argv) == 3
    individual_file = sys.argv[1]
    location_file = sys.argv[2]
    main(individual_file, location_file)

