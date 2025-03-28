import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

EU27_LIST = {
    'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Denmark', 'Estonia', 'Finland',
    'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg',
    'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden'
}

RED3_INDUSTRY_TARGETS = {2030: 0.42, 2035: 0.6}  # RED3 targets
STOICHIOMETRIC_RATIOS = {"Ammonia": 0.182, "Methanol": 0.128}  # how many tons of H2 are required for a ton of ammonia or methanol

# from https://www.engineeringtoolbox.com/fuels-higher-calorific-values-d_169.html
LHV_VALUES = {"Methanol": 19.9, "Jet fuel": 43.1}  # Lower heating values in GJ/ton
GWH_TO_GJ = 3600

def clean_industry_data(df):
    """Cleans industry data

    Args:
        df: industry dataframe

    Returns:
        df: cleaned industry dataframe
    """
    df = df[['Country', 'country filters', 'End-use', 'Total consumption']].copy()
    df.columns = ['Country', 'Country Type', 'End-use', 'Total consumption']
    df = df.pivot_table(index = ['Country', 'Country Type'], columns = 'End-use', values = 'Total consumption').fillna(0)
    return df.reset_index()

def clean_transport_data(df):
    """Cleans transport data

    Args:
        df: transport dataframe

    Returns:
        df: cleaned transport dataframe
    """
    df.columns = ['Country', 'International maritime', 'International aviation', 'Domestic aviation', 'Domestic maritime']
    df.replace('European Union - 27 countries (from 2020)', 'EU27 Total', inplace = True)
    return df[df['Country'] != 'EU27 Total'].replace(':', 0).copy()

def apply_stoichiometric_conversion(df):
    """Apply stoichiometric conversion to the industry dataframe.
    Converts tons of hydrogen to tons of ammonia and methanol.

    Args:
        df: Industry dataframe

    Returns:
       df: df with stoichiometric conversion applied
    """
    for fuel, ratio in STOICHIOMETRIC_RATIOS.items():
        df[fuel] /= ratio
    return df

def apply_red3_targets(df, target_year):
    """Calculate RED3 targets for the industry sector.

    Args:
        df: Industry dataframe
        target_year: Year for RED3 target calculations

    Returns:
        df: df with RED3 industry targets applied
    """
    factor = RED3_INDUSTRY_TARGETS[target_year]
    for col in ['Ammonia', 'Methanol', 'H2']:
        df[col] *= factor
    return df

def process_industry_data(df, target_year):
    """Process industry data to calculate the demand for green liquids according to RED3 directive.

    Args:
        df: Industry dataframe
        target_year: Year for RED3 target calculations

    Returns:
        df: Dataframe with target demand for green liquids in industry sector according to RED3 directive.
    """
    # remove sectors that are not relevant: e-fuels and steel (very small applications)
    # as well as other chemicals (since we exclude the chemical sector) and "others"
    df = df[['Country', 'Country Type', 'Ammonia', 'Industrial heat', 'Methanol']].copy()

    df.rename(columns={'Industrial heat': 'H2'}, inplace=True)
    df = apply_stoichiometric_conversion(df) # convert from tons of hydrogen to tons of ammonia and tons of methanol
    df = apply_red3_targets(df, target_year) # apply RED3 target to derive green liquids demand
    df['sector'] = 'industry'
    return df.melt(id_vars=['Country', 'Country Type', 'sector'], var_name='fuel', value_name='amount')

def process_transport_data(df):
    """Process transport data to calculate the demand for green liquids according to RED3 directive.

    Args:
        df: Transport dataframe

    Returns:
        df_target: Dataframe with target demand for green liquids in transport sector according to RED3 directive.
    """
    df = df.copy()
    df['International maritime'] *= GWH_TO_GJ / LHV_VALUES['Methanol']
    df['Domestic maritime'] *= GWH_TO_GJ / LHV_VALUES['Methanol']
    df['International aviation'] *= GWH_TO_GJ / LHV_VALUES['Jet fuel']
    df['Domestic aviation'] *= GWH_TO_GJ / LHV_VALUES['Jet fuel']
    df['Jet fuel'] = df['International aviation'] + df['Domestic aviation']
    df['Methanol'] = df['International maritime'] + df['Domestic maritime']
    df_target = df[['Country', 'Jet fuel', 'Methanol']].copy()
    # Apply 2030 RED3 targets 
    # The same are used for 2035 as there are no further targets until 2050
    df_target['Jet fuel'] *= 0.012
    df_target['Methanol'] *= 0.01
    df_target['sector'] = 'transport'
    return df_target.melt(id_vars=['Country', 'sector'], var_name='fuel', value_name='amount')

def analyse_eu_greenliquid_targets(industry_df, transport_df, target_year = 2030):
    """
    Calculates EU green liquid targets for industry and transport sectors according to the RED3 directive.
    
    Parameters:
    - industry_df: DataFrame containing industry consumption of hydrogen.
    - transport_df: DataFrame containing transport energy demand data.
    - target_year: Year for RED3 target calculations (default: 2030).
    
    Returns:
    - DataFrame containing the processed green liquid fuel targets for the EU.
    """
    if target_year not in RED3_INDUSTRY_TARGETS:
        raise ValueError("Target year not supported. Choose 2030 or 2035.")

    industry_df = process_industry_data(industry_df, target_year)
    transport_df_target = process_transport_data(transport_df)

    df = pd.concat([industry_df, transport_df_target], axis = 0).reset_index(drop = True)
    df['Country Type'] = df['Country'].apply(lambda x: 'EU27' if x in EU27_LIST else 'Other')

    return df[df['Country Type'] == 'EU27'].reset_index(drop = True)
    

def plot_eu_liquids(industry_df, transport_df, target_year=2030):
    """Plot EU green liquid targets according to RED3 directive for industry and transport sectors.

    Args:
        industry_df: cleaned industry data
        transport_df: cleaned transport data
        target_year (int, optional): Year to use for the target. Defaults to 2030.

    Returns:
        df: full DataFrame with the processed green liquid fuel targets for the EU.
    """
    df = analyse_eu_greenliquid_targets(industry_df, transport_df, target_year=target_year)
    df = df[df['fuel'] != 'H2']
    df['amount'] = df['amount'] / 1e3 # convert to kt

    total_demand = df.groupby(['fuel', 'sector'])['amount'].sum().reset_index()
    total_demand['Country'], total_demand['Country Type'] = 'Total', 'EU27'
    df = pd.concat([df, total_demand], ignore_index=True)

    df['Country Group'] = df['Country'].apply(lambda x: 'Rest of Europe' if x not in ['France', 'Germany', 'Belgium', 'Netherlands', 'Total', 'Spain', 'Poland'] else x)
    df_aggregated = df.groupby(['fuel', 'sector', 'Country Group', 'Country Type'])['amount'].sum().reset_index()

    country_order = ['Total','Netherlands', 'Germany', 'Belgium','France','Spain','Poland', 'Rest of Europe']
    df_aggregated['Country Group'] = pd.Categorical(df_aggregated['Country Group'], categories=country_order, ordered=True)

    fuels = df_aggregated['fuel'].unique()
    colors = {'industry': 'skyblue', 'transport': 'salmon'}
    fig, axes = plt.subplots(nrows=len(fuels), ncols=1, figsize=(10, 18), sharex=True)
    fig.suptitle(f'Demand for green liquids in the EU in {target_year}', fontsize = 16, weight = 'bold')

    for ax, fuel in zip(axes, fuels):
        fuel_data = df_aggregated[df_aggregated['fuel'] == fuel].pivot_table(index=['Country Group'], columns='sector', values='amount', fill_value=0)
        fuel_data.plot(kind='bar', stacked=True, ax=ax, color=[colors[sector] for sector in fuel_data.columns])

        total_heights = fuel_data.sum(axis=1)
        for i, (index, total_height) in enumerate(total_heights.items()):
            ax.annotate(f'{total_height:.2f}',
                        (i, total_height),
                        ha='center', va='bottom',
                        fontsize=10, color='black',
                        xytext=(0, 0), textcoords='offset points')
        ax.set_title(f'Demand for green {fuel} by country and sector')
        ax.set_xlabel('Country')
        ax.set_ylabel('Amount (kt)')
        ax.legend(title='Sector')

    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.savefig(f'output/EU_green_liquids_{target_year}.png')

    return df

# load and process data
EU_industry_df = clean_industry_data(pd.read_excel('inputdata/H2EU_Industry_consumption_data.xlsx'))
EU_transport_df = clean_transport_data(pd.read_excel('inputdata/EU_Transportenergy_data.xlsx', sheet_name = "Sheet 1"))

df_2030 = plot_eu_liquids(EU_industry_df, EU_transport_df, target_year=2030)
df_2035 = plot_eu_liquids(EU_industry_df, EU_transport_df, target_year=2035)
full_df = pd.concat([df_2030.assign(year=2030), df_2035.assign(year=2035)], axis=0)

full_df.to_csv('output/EU_green_liquids_RED3Targets.csv', index = False)
