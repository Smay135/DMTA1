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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



df = pd.read_csv('/Users/clairek/Desktop/Master/dataset_mood_smartphone.csv',delimiter=',')
df.info()
df.head()
df.columns

# (number of) participants
participants = df['id'].unique()
#print(participants)
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
original_df = df.copy()
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
df = original_df
# DO THE SAME THING BUT THEN WITH SPECIFIC TIMEPOINTS IN A DAY !!
# DO THE SAME THING BUT THEN WITH FREQUENCY OF ANSWERS TO SPECIFIC FEATURES !!
'''did all participants start at different times ??'''




'''add plots of features: i.e., pd filter on feature type and plot values'''

# set index as different time (this has to come after the plots bc of indexing)
#df.set_index('time', inplace=True)
# df.set_index('time')
# df.loc['AS.14.01]
# df.set_index('time').sort_index(inplace=True)


### filtering data ###
feature_library = {}

# filter function
def feature_filter(feature):
    feature_df = df.loc[df['variable']==feature]
    #print(f"features={feature_df}")
    return feature_df

# storing dfs for all features in library
for feature in features:
    feature_library[feature] = feature_filter(feature)

# access df of each feature by changing variable name
variables = df['variable'].unique()
print(variables)

var_df_lib = {}
for var in variables:
    var_df_lib[var] = feature_library[var]

print(f"*****DICTIONARY *****{var_df_lib}")
# mood_df = feature_library['mood']
# arous_df = feature_library['circumplex.arousal']
# valen_df = feature_library['circumplex.valence']
# activity_df = feature_library['activity']
# screen_df = feature_library['screen']
# builtin_df = feature_library['appCat.builtin']
# comm_df = feature_library['appCat.communication']
# ent_df = feature_library['appCat.entertainment']
# finance_df = feature_library['appCat.finance']
# game_df = feature_library['appCat.game']
# office_df = feature_library['appCat.office']
# other_df = feature_library['appCat.other']
# social_df = feature_library['appCat.social']
# travel_df = feature_library['appCat.social']
# unk_df = feature_library['appCat.unknown']
# util_df = feature_library['appCat.utilities']
# weather_df =feature_library['appCat.weather']

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


#sns.boxplot(feature_counts).set_title('variables')
#plt.show()



# descriptive statistics of individual features


# correlation matrix



'''show daily trends with dips in time (night & early morning)'''


'''aggregate here and analyse again'''




# data cleaning
# 1. take out outliers

def mood_out(var_df, ppl):
    filtered_rows_list = []  # List to collect filtered DataFrames for each 'id'
    for p in ppl:
        # Filter var_df to get rows where 'id' matches the current 'p'
        filtered_rows = var_df[var_df['id'] == p].copy()  # Make a copy of the filtered DataFrame

        lower = 1
        upper = 10
        filtered_rows.loc[filtered_rows['value'] < lower, 'value'] = 1
        filtered_rows.loc[filtered_rows['value'] > upper, 'value'] = 10

        filtered_rows_list.append(filtered_rows)
    result_df = pd.concat(filtered_rows_list, ignore_index=True)

    return result_df


def out_pp(var_df, ppl):
    filtered_rows_list = []  # List to collect filtered DataFrames for each 'id'

    for p in ppl:
        # Filter var_df to get rows where 'id' matches the current 'p'
        filtered_rows = var_df[var_df['id'] == p].copy()  # Make a copy of the filtered DataFrame

        # Calculate quartiles and IQR for outlier detection
        Q1 = filtered_rows['value'].quantile(0.25)
        Q3 = filtered_rows['value'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Cap outliers to lower and upper bounds
        filtered_rows.loc[filtered_rows['value'] < lower, 'value'] = lower
        filtered_rows.loc[filtered_rows['value'] > upper, 'value'] = upper

        # Append filtered_rows to the list
        filtered_rows_list.append(filtered_rows)

    # Concatenate all filtered DataFrames in the list
    result_df = pd.concat(filtered_rows_list, ignore_index=True)

    # Print remaining percentage of rows in the original DataFrame
    print("Percentage of rows remaining:")
    print(percent_left(var_df, result_df))

    return result_df

def percent_left(old, new):
    percent = len(new)/len(old)
    percent = percent*100
    return percent

# Dictionary to hold the output DataFrames
olf_dataframes = {}

# Loop through each DataFrame and apply out_pp function
for df_name, df in var_df_lib.items():
    olf_dataframes[df_name] = out_pp(df, participants)

olf_df_mood = mood_out(var_df_lib['mood'], participants)
olf_dataframes = {'mood_df': olf_df_mood, **olf_dataframes}
print(olf_dataframes)

#sns.boxplot(data = olf_df_mood, x='id', y= 'value')
#plt.show()
# 2. missing values solution: average out values -- aggregate per day



#averages out per person per day period (before noon (am), after noon (pm))
def avg_time(var_df, ppl):
    result_list = {}  # List to collect daily averages for each participant

    for p in ppl:
        # Filter var_df to get rows where 'id' matches the current 'p'
        filtered_rows = var_df[var_df['id'] == p].copy()
        #print(f"result_list: {result_list}")
        # Convert 'time' column to datetime
        filtered_rows['time'] = pd.to_datetime(filtered_rows['time'])

        # Extract day, hour, and time of day
        filtered_rows['day'] = filtered_rows['time'].dt.date
        filtered_rows['hour'] = filtered_rows['time'].dt.hour
        filtered_rows['time_of_day'] = filtered_rows['hour'].apply(lambda x: 'am' if x < 12 else 'pm')

        daily_averages = filtered_rows.groupby(['id', 'day', 'time_of_day'])['value'].mean().unstack()
        print(daily_averages)
        # Append daily averages to result_list
        result_list[p] = daily_averages
        #result_list.append(daily_averages)


    return result_list

olf_time_dataframes = {}

# Loop through each DataFrame and apply out_pp function
for df_name, df in olf_dataframes.items():
    olf_time_dataframes[df_name] = avg_time(df, participants)

print(olf_time_dataframes)



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
