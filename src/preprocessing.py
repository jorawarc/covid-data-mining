
import os
import re
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
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
    #date_time_df = pd.to_datetime(individual_df['date_confirmation'])
    #date_time_no_outliers_df = remove_outliers_zscore(date_time_df)
    confirmed_no_outliers_df = remove_outliers_zscore(location_df['Confirmed'])
    deaths_no_outliers_df = remove_outliers_zscore(location_df['Deaths'])
    recovered_no_outliers_df = remove_outliers_zscore(location_df['Recovered'])
    active_no_outliers_df = remove_outliers_zscore(location_df['Active'])
    incidence_rate_no_outliers_df = remove_outliers_zscore(location_df['Incidence_Rate'])
    case_fatality_ratio_no_outliers_df = remove_outliers_zscore(location_df['Case-Fatality_Ratio'])

    # Print count of outliers
    #print(date_time_no_outliers_df)
    #print(confirmed_no_outliers_df)
    #print(deaths_no_outliers_df)
    #print(recovered_no_outliers_df)
    #print(active_no_outliers_df)
    #print(incidence_rate_no_outliers_df)

    print()
    print("case_fatality_ratio_no_outliers_df")
    print(case_fatality_ratio_no_outliers_df)

def remove_outliers_zscore(S):
    print(type(S))
    S = S[~((S-S.mean()).abs() > 3*S.std())]
    return S

def remove_outliers_iqr(df):
    quartiles = np.nanpercentile(df, [25, 75])
    first_quartile = quartiles[0]
    third_quartile = quartiles[1]
    # Calc inter quartile range
    inter_quartile_range = third_quartile - first_quartile
    # Calc outlier bounds
    lower_bound = first_quartile - inter_quartile_range * 1.5
    upper_bound = third_quartile + inter_quartile_range * 1.5
    # Filter df to get outliers
    no_outliers_df = df[(df < upper_bound) & (df > lower_bound)]

    return no_outliers_df

def transform(df):
    US_df = df[df['Country_Region'] == 'US']
    country_state_df = US_df.groupby(['Province_State']).agg({'Confirmed': ['count']})
    print(country_state_df)

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

    transformed_df = transform(location_df)

def main1(individual_file, location_file):
    individual_df = pd.read_csv(individual_file)
    outliers = detect_outliers(individual_df)

if __name__ == '__main__':
    assert len(sys.argv) == 3
    individual_file = sys.argv[1]
    location_file = sys.argv[2]
    main(individual_file, location_file)

