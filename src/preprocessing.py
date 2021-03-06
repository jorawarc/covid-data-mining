import os
import re
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

FIGURE_DIR = '../figures'
DATA_DIR = '../data'
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
    confirmed = country_df[country_df['Confirmed'] > 200000].sort_values('Confirmed')
    recovered = country_df[country_df['Recovered'] > 50000].sort_values('Recovered')

    save_barh_figure(deaths.index, deaths['Deaths'], 'Deaths by Country', 'Number of Deaths',
                     'Country', os.path.join(FIGURE_DIR, 'deaths_by_country.png'))
    save_barh_figure(confirmed.index, confirmed['Confirmed'], 'Confirmed Cases by Country',
                     'Number of Cases', 'Country', os.path.join(FIGURE_DIR, 'cases_by_country.png'))
    save_barh_figure(recovered.index, recovered['Recovered'], 'Recoveries by Country',
                     'Number of Recoveries', 'Country', os.path.join(FIGURE_DIR, 'recoveries_by_country.png'))


def visual_histograms(df, is_categorical=False):
    if is_categorical:
        for col in df.columns:
            data = df.groupby([col]).size().reset_index(name="count").sort_values('count', ascending=False).head(20)
            chart = sns.barplot(x=col, y='count', data=data, palette='rocket')
            chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=7)
            plt.title('{} Frequency'.format(col[0].upper() + col[1:]))
            plt.show()
    else:
        for col in df.columns:
            plt.figure()
            sns.histplot(df[col], color="skyblue")
            plt.title('{} Frequency'.format(col[0].upper() + col[1:]))
        plt.show()


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


def remove_outliers_individual_df(individual_df):
    age_outliers_df = get_outliers_zscore(individual_df, individual_df['age'])
    individual_df['epoch_date_confirmation'] = (individual_df['date_confirmation'] - pd.Timestamp(
        "1970-01-01")) // pd.Timedelta('1s')

    date_time_outliers_df = get_outliers_zscore(individual_df, individual_df['epoch_date_confirmation'])
    outliers_count_df = pd.DataFrame({"attribute": ['age', 'date_confirmation'],
                                      "outliers_count": [age_outliers_df['age'].count(),
                                                         date_time_outliers_df['epoch_date_confirmation'].count()]})
    print("Individual Cases DF:")
    print(outliers_count_df)

    individual_df = individual_df.merge(age_outliers_df, indicator=True, how='left').loc[
        lambda x: x['_merge'] != 'both'].drop('_merge', axis=1)

    return individual_df


def print_outliers_count_location_df(location_df):
    # get all outliers in their own df
    confirmed_outliers_df = get_outliers_zscore(location_df, location_df['Confirmed'])
    deaths_outliers_df = get_outliers_zscore(location_df, location_df['Deaths'])
    recovered_outliers_df = get_outliers_zscore(location_df, location_df['Recovered'])
    active_outliers_df = get_outliers_zscore(location_df, location_df['Active'])
    incidence_rate_outliers_df = get_outliers_zscore(location_df, location_df['Incidence_Rate'])
    case_fatality_ratio_outliers_df = get_outliers_zscore(location_df, location_df['Case-Fatality_Ratio'])

    outliers_count = []
    outliers_count.append(["Confirmed", confirmed_outliers_df.shape[0]])
    outliers_count.append(["Deaths", deaths_outliers_df.shape[0]])
    outliers_count.append(["Recovered", recovered_outliers_df.shape[0]])
    outliers_count.append(["Active", active_outliers_df.shape[0]])
    outliers_count.append(["Incidence_rate", incidence_rate_outliers_df.shape[0]])
    outliers_count.append(["Case_fatality_ratio", case_fatality_ratio_outliers_df.shape[0]])

    outliers_count_df = pd.DataFrame(outliers_count, columns=['attribute', 'outliers_count'])

    print("Location DF:")
    print(outliers_count_df)


def remove_outliers_location_df(location_df):
    # get all outliers in their own df
    confirmed_outliers_df = get_outliers_zscore(location_df, location_df['Confirmed'])
    deaths_outliers_df = get_outliers_zscore(location_df, location_df['Deaths'])
    recovered_outliers_df = get_outliers_zscore(location_df, location_df['Recovered'])
    active_outliers_df = get_outliers_zscore(location_df, location_df['Active'])
    incidence_rate_outliers_df = get_outliers_zscore(location_df, location_df['Incidence_Rate'])
    case_fatality_ratio_outliers_df = get_outliers_zscore(location_df, location_df['Case-Fatality_Ratio'])

    print(case_fatality_ratio_outliers_df)

    # join on set complement (or difference)
    print("before merge location_df ")
    location_df = location_df.merge(confirmed_outliers_df, indicator=True, how='left').loc[
        lambda x: x['_merge'] != 'both'].drop('_merge', axis=1)
    location_df = location_df.merge(deaths_outliers_df, indicator=True, how='left').loc[
        lambda x: x['_merge'] != 'both'].drop('_merge', axis=1)
    location_df = location_df.merge(recovered_outliers_df, indicator=True, how='left').loc[
        lambda x: x['_merge'] != 'both'].drop('_merge', axis=1)
    location_df = location_df.merge(active_outliers_df, indicator=True, how='left').loc[
        lambda x: x['_merge'] != 'both'].drop('_merge', axis=1)
    location_df = location_df.merge(incidence_rate_outliers_df, indicator=True, how='left').loc[
        lambda x: x['_merge'] != 'both'].drop('_merge', axis=1)
    location_df = location_df.merge(case_fatality_ratio_outliers_df, indicator=True, how='left').loc[
        lambda x: x['_merge'] != 'both'].drop('_merge', axis=1)
    return location_df


def get_outliers_zscore(df, s):
    # df = df[np.abs(s - s.mean()) <= (3 * s.std())]
    # keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.

    outliers_df = df[(np.abs(s - s.mean()) > (3 * s.std()))]
    # or if you prefer the other way around
    return outliers_df


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


def aggregate_US_counties(df):
    US_df = df[df['Country_Region'] == 'US']
    country_state_df = US_df[['Province_State', 'Confirmed', 'Deaths', 'Recovered', 'Active', 'Incidence_Rate', 'Case-Fatality_Ratio']].groupby('Province_State').sum()
    country_state_df['Case-Fatality_Ratio'] = country_state_df['Deaths'] / country_state_df['Confirmed'] * 100
    print("=== USA Aggregation ===")
    print(country_state_df)
    return country_state_df


def join_data_sets(individual_df, location_df):
    merged_df = individual_df.merge(right=location_df, left_on=['country', 'province'],
                                    right_on=['Country_Region', 'Province_State'],
                                    indicator=True, how='left')
    merged_df = merged_df[merged_df['_merge'] == 'both']
    merged_df = merged_df.drop(columns=['_merge'])
    print('=== Merged Data ===')
    print(merged_df)
    return merged_df

  
def main(individual_file, location_file):
    individual_df = pd.read_csv(individual_file)
    individual_df['age'] = individual_df['age'].apply(reduce_age_range)
    individual_df['date_confirmation'] = individual_df['date_confirmation'].apply(reduce_date_confirmation)
    individual_df['date_confirmation'] = pd.to_datetime(individual_df['date_confirmation'])

    location_df = pd.read_csv(location_file, parse_dates=['Last_Update'])

    print_stats(individual_df, location_df)
    print_missing(individual_df, location_df)

    # Handle Case where Location data: NaN, Country Individual data: province, country
    # Instead find these miss matched cases and set the province value in the Individual data to 'unknown'
    missing_provinces = location_df[pd.isna(location_df['Province_State'])]
    missing_on_merge = missing_provinces.merge(individual_df, left_on='Country_Region', right_on='country', indicator=True)
    missing_on_merge = missing_on_merge[(~pd.isna(missing_on_merge['province']) & (pd.isna(missing_on_merge['Province_State'])))]
    countries = list(missing_on_merge['Country_Region'].unique())

    removal_countries = location_df[location_df['Country_Region'].isin(countries)].groupby('Country_Region').size().to_frame('size').reset_index()
    removal_countries = removal_countries[removal_countries['size'] == 1]

    individual_df.loc[individual_df['country'].isin(removal_countries['Country_Region']), 'province'] = 'unknown'

    # Impute Age, Sex, Province for Individual DF / Province, Case Fatality for Location DF
    individual_df[['sex', 'province']] = individual_df[['sex', 'province']].fillna(value="unknown")
    individual_df['age'] = individual_df['age'].astype(np.float)
    # individual_df['age'] = individual_df['age'].fillna(value=individual_df['age'].sum()/individual_df['age'].count())
    location_df['Province_State'] = location_df['Province_State'].fillna(value='unknown')

    # Drop missing Lat, Long columns
    location_df.dropna(subset=['Lat', 'Long_'], inplace=True)
    individual_df.dropna(subset=['latitude', 'longitude'], inplace=True)

    location_df['Case-Fatality_Ratio'] = location_df['Case-Fatality_Ratio'].fillna(value=0)
    individual_df['country'] = individual_df['country'].fillna(value="Taiwan")
    print("=== Changes After Imputation Process ===")
    print_missing(individual_df, location_df)

    # Generate Visuals
    generate_visuals(individual_df, location_df)

    print("=== Outliers ===")
    individual_df = remove_outliers_individual_df(individual_df)
    print_outliers_count_location_df(location_df)

    country_state_df = aggregate_US_counties(location_df)
    country_state_df.to_csv(os.path.join(DATA_DIR, 'USA_Aggregated_Data.csv'))

    merged_df = join_data_sets(individual_df, location_df)
    merged_df.to_csv(os.path.join(DATA_DIR, 'Merged_Data_Sets.csv'), index=False)


def generate_visuals(individual_df, location_df):
    visual_by_country(location_df)
    visual_histograms(
        location_df[['Confirmed', 'Deaths', 'Recovered', 'Active', 'Incidence_Rate', 'Case-Fatality_Ratio']],
        is_categorical=False)
    visual_histograms(location_df[['Province_State', 'Country_Region']], is_categorical=True)
    visual_histograms(individual_df[['sex', 'outcome']], is_categorical=True)
    visual_histograms(individual_df[['age']][individual_df['age'] != 'unknown'].astype(np.float), is_categorical=False)


def print_missing(individual_df, location_df):
    print("=== Missing Values ===")
    print("Individual Cases DF:")
    print(compute_missing_values(individual_df))
    print("Location DF:")
    print(compute_missing_values(location_df))


def print_stats(individual_df, location_df):
    print("=== Stats ===")
    print("Individual Cases DF:")
    print(individual_df[['age']].describe())
    print("Location DF:")
    print(location_df[['Confirmed', 'Deaths', 'Recovered', 'Active']].describe())
    print(location_df[['Incidence_Rate', 'Case-Fatality_Ratio']].describe())


if __name__ == '__main__':
    assert len(sys.argv) == 3
    individual_file = sys.argv[1]
    location_file = sys.argv[2]
    main(individual_file, location_file)