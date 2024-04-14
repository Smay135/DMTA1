#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 17:07:04 2024

@author: shivanikandhai

DMT assignment 1
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



df = pd.read_csv('/Users/clairek/Desktop/Master/dataset_mood_smartphone.csv',delimiter=',')
df.info()
df.head()
df.columns

# (number of) participants
participants = df['id'].unique()
n = len(participants)

# (number of) features
features = df['variable'].unique()
nr_features = len(features) 

# time to datetime
df['time'] = pd.to_datetime(df['time'])


# PLOTTING FEATURES 
# datapoints per participant (= unequal)
pc = df['id'].value_counts()
participant_counts = pd.DataFrame(pc)
#print(participant_counts)
# plotting logging data of individual participants
# plt.figure(figsize=(10,6))
# #had to change the participant_counts['id'] to participant_counts.iloc[:, 0] bc i kept getting errors - same with time at ln 52
# plt.vlines(x=participant_counts.index, ymin=0, ymax=participant_counts.iloc[:, 0], color='blue')
# plt.title('Datapoints per Participant')
# plt.xlabel('Participant ID')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.xticks(rotation=45) # rotate x-axis labels to prevent overlap
# plt.tight_layout()
#plt.show()
# participant_counts.boxplot()
# plt.show()
sns.boxplot(participant_counts).set_title('participants')
#plt.show()

# frequency of time logs (shows bias) 
time_counts = pd.DataFrame(df['time'].value_counts())
#print(time_counts)
#plotting the frequency data
plt.figure(figsize=(10,6))
plt.vlines(x=time_counts.index, ymin=0, ymax=time_counts.iloc[:, 0], color='blue')
#plt.plot(time_counts.index, time_counts['time'], marker='o')
plt.title('Frequency over Time')
plt.xlabel('Date and Time')
plt.ylabel('Frequency')
plt.grid(True)
#plt.xticks(rotation=45) # rotate x-axis labels
plt.tight_layout()
#plt.show()

# DO THE SAME THING BUT THEN WITH SPECIFIC TIMEPOINTS IN A DAY !!
# DO THE SAME THING BUT THEN WITH FREQUENCY OF ANSWERS TO SPECIFIC FEATURES !!
'''did all participants start at different times ??'''




'''add plots of features: i.e., pd filter on feature type and plot values'''

# set index as different time (this has to come after the plots bc of indexing)
df.set_index('time', inplace=True)
# df.set_index('time')
# df.loc['AS.14.01]
# df.set_index('time').sort_index(inplace=True)


### filtering data ###
feature_library = {}

# filter function
def feature_filter(feature):
    feature_df = df.loc[df['variable']==feature]
    return feature_df

# storing dfs for all features in library
for feature in features:
    feature_library[feature] = feature_filter(feature)

# access df of each feature by changing variable name
variables = df['variable'].unique()
print(variables)

mood_df = feature_library['mood']
arous_df = feature_library['circumplex.arousal']
valen_df = feature_library['circumplex.valence']
activity_df = feature_library['activity']
screen_df = feature_library['screen']
builtin_df = feature_library['appCat.builtin']
comm_df = feature_library['appCat.communication']
ent_df = feature_library['appCat.entertainment']
finance_df = feature_library['appCat.finance']
game_df = feature_library['appCat.game']
office_df = feature_library['appCat.office']
other_df = feature_library['appCat.other']
social_df = feature_library['appCat.social']
travel_df = feature_library['appCat.social']
unk_df = feature_library['appCat.unknown']
util_df = feature_library['appCat.utilities']
weather_df =feature_library['appCat.weather']

feature_df = feature_library['mood']
#print(feature_df)
sns.boxplot(feature_df['value']).set_title('values')
#plt.show()
# further indexing possible for prediction etc.: feature_library['mood']['value'][3]

# frequency of feature logs (bring to log scale ??)
feature_counts = pd.DataFrame(df['variable'].value_counts())
# plotting logging data of individual features
# plt.figure(figsize=(10,6))
# plt.vlines(x=feature_counts.index, ymin=0, ymax=feature_counts.iloc[:, 0], color='blue')
# plt.title('Datapoints per Feature')
# plt.xlabel('Feature')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.xticks(rotation=45) # rotate x-axis labels to prevent overlap
# plt.tight_layout()
#plt.show()
# feature_counts.boxplot()
# plt.show()

#used boxplot to find outliers
#anything over 40000
#sns.boxplot(feature_counts).set_title('variables')
#plt.show()



# descriptive statistics of individual features


# correlation matrix



'''show daily trends with dips in time (night & early morning)'''


'''aggregate here and analyse again'''




# data cleaning
# 1. take out outliers
def removal_box_plot(d_f, column):
    sns.boxplot(data = d_f, x='id', y= 'value')
    #sns.boxplot(data=titanic, x="class", y="age", hue="alive")
    plt.title(f'Original Box Plot of {column}')
    plt.show()

    # removed_outliers = d_f[d_f[column] <= threshold]
    #
    # sns.boxplot(removed_outliers[column])
    # plt.title(f'Box Plot without Outliers of {column}')
    # plt.show()
    # return removed_outliers


# threshold_value =

no_outliers = removal_box_plot(feature_df, 'values')

def outliers_pp(var_df, ppl):
    # Create an empty DataFrame to store appended rows
    olf_df = pd.DataFrame(columns=var_df.columns)  # Assuming var_df has columns you want to preserve

    # Iterate over each 'id' in ppl
    for p in ppl:
        # Filter var_df to get rows where 'id' matches the current 'p'
        filtered_rows = var_df[var_df['id'] == p].copy()  # Make a copy of the filtered DataFrame

        # Calculate quartiles and IQR for outlier detection
        Q1 = filtered_rows['value'].quantile(0.25)
        Q3 = filtered_rows['value'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Identify outlier rows based on value column
        outliers_mask = (filtered_rows['value'] < lower) | (filtered_rows['value'] > upper)

        # Remove outliers
        filtered_rows = filtered_rows[~outliers_mask]

        # Append filtered rows to olf_df
        olf_df = pd.concat([olf_df, filtered_rows], ignore_index=True)

    return olf_df



olf_df_mood = outliers_pp(mood_df, participants)
olf_df_arous = outliers_pp(arous_df, participants)
olf_df_val = outliers_pp(valen_df, participants)
olf_df_act = outliers_pp(activity_df, participants)
olf_df_scr = outliers_pp(screen_df, participants)
olf_df_bi = outliers_pp(builtin_df, participants)
olf_df_comm = outliers_pp(comm_df, participants)
olf_df_ent = outliers_pp(ent_df, participants)
olf_df_fin = outliers_pp(finance_df, participants)
olf_df_game = outliers_pp(game_df, participants)
olf_df_off = outliers_pp(office_df, participants)
olf_df_oth = outliers_pp(other_df, participants)
olf_df_soc = outliers_pp(social_df, participants)
olf_df_trav = outliers_pp(travel_df, participants)
olf_df_unk = outliers_pp(unk_df, participants)
olf_df_util = outliers_pp(util_df, participants)
olf_df_weath = outliers_pp(weather_df, participants)

sns.boxplot(data = olf_df_mood, x='id', y= 'value')
plt.show()
# 2. missing values solution: average out values -- aggregate per day 

''' consider what to do with prolonged periods of missing values''' # use average/median value
''' are time points the same for each participant''' # no -- prove this !




















##################

### EXAMPLE CODE TIME SERIES PLOT ###
# # Example of plotting the time series data for one variable for each participant
# for participant_id in df['id'].unique():
#     participant_data = df[df['id'] == participant_id] # FILTER HERE
#
#     # Let's assume you want to analyze the first variable 'var1'
#     plt.figure(figsize=(10, 5))
#     plt.plot(participant_data.index, participant_data.iloc[:, 0], label=f'Participant {participant_id}')
#     plt.title(f'Time Series Plot for Participant {participant_id} - Variable 1')
#     plt.xlabel('Date')
#     plt.ylabel('Value of Variable 1')
#     plt.legend()
    #plt.show()
    #participant_data.boxplot()



# ##################
#
# # selecting rows by index
# df.iloc[0]
# df.iloc[[0,1]] # df.iloc[[0,1],1] for specific cell
#
# # indexing colums/series
# df[['id','time']]
# df['id']
#
# # df.loc() = location without index ~ can use names of columns
# df.loc[0,'time']
# df.loc[[0,1,2],'time'] # or df.loc[0:2,'time'] ~ slicing is inclusive of all mentioned values
#
#
#
#
# ppl = {
#        'first': ['corey','jane','john'],
#        'last': ['smith','doe','doe'],
#        'email': ['cc@gmail.com','janedoe@gmail.com','johndoe@gmail.com']
#        }
#
# for i in ppl.keys():
#     print(ppl[i][0])
#
# pd.DataFrame(ppl)

# ##################
#
# # Step 1: Create dummy data
# # Generate sample data
# np.random.seed(0)
# dates = pd.date_range('20230101', periods=100)
# data = pd.DataFrame({
#     'Person1': np.random.randn(100).cumsum(),
#     'Person2': np.random.randn(100).cumsum()
# }, index=dates)
#
# # Step 2: Calculate Pearson correlation
# pearson_corr = data['Person1'].corr(data['Person2'])
# print(f"Pearson correlation: {pearson_corr}")
#
# # Step 3: Calculate Spearman correlation
# spearman_corr = data['Person1'].corr(data['Person2'], method='spearman')
# print(f"Spearman correlation: {spearman_corr}")
