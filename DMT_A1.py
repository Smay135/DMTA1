#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 17:07:04 2024

@author: shivanikandhai

DMT assignment 1
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/shivanikandhai/Documents/School/Artificial_Intelligence/Data Mining/dataset_mood_smartphone.csv',delimiter=',')
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
participant_counts = pd.DataFrame(df['id'].value_counts())
# plotting logging data of individual participants
plt.figure(figsize=(10,6))
plt.vlines(x=participant_counts.index, ymin=0, ymax=participant_counts['id'], color='blue')
plt.title('Datapoints per Participant')
plt.xlabel('Participant ID')
plt.ylabel('Frequency')
plt.grid(True)
plt.xticks(rotation=45) # rotate x-axis labels to prevent overlap
plt.tight_layout() 
plt.show()

# frequency of time logs (shows bias) 
time_counts = pd.DataFrame(df['time'].value_counts())
# plotting the frequency data
plt.figure(figsize=(10,6))
plt.vlines(x=time_counts.index, ymin=0, ymax=time_counts['time'], color='blue')
#plt.plot(time_counts.index, time_counts['time'], marker='o')
plt.title('Frequency over Time')
plt.xlabel('Date and Time')
plt.ylabel('Frequency')
plt.grid(True)
#plt.xticks(rotation=45) # rotate x-axis labels
plt.tight_layout() 
plt.show()
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
feature_df = feature_library['mood']
# further indexing possible for prediction etc.: feature_library['mood']['value'][3]

# frequency of feature logs (bring to log scale ??)
feature_counts = pd.DataFrame(df['variable'].value_counts())
# plotting logging data of individual features
plt.figure(figsize=(10,6))
plt.vlines(x=feature_counts.index, ymin=0, ymax=feature_counts['variable'], color='blue')
plt.title('Datapoints per Feature')
plt.xlabel('Feature')
plt.ylabel('Frequency')
plt.grid(True)
plt.xticks(rotation=45) # rotate x-axis labels to prevent overlap
plt.tight_layout() 
plt.show()


# descriptive statistics of individual features


# correlation matrix



'''show daily trends with dips in time (night & early morning)'''


'''aggregate here and analyse again'''




# data cleaning
# 1. take out outliers
# 2. missing values solution: average out values -- aggregate per day 

''' consider what to do with prolonged periods of missing values''' # use average/median value
''' are time points the same for each participant''' # no -- prove this !




















##################

### EXAMPLE CODE TIME SERIES PLOT ###
# Example of plotting the time series data for one variable for each participant
for participant_id in df['id'].unique():
    participant_data = df[df['id'] == participant_id] # FILTER HERE

    # Let's assume you want to analyze the first variable 'var1'
    plt.figure(figsize=(10, 5))
    plt.plot(participant_data.index, participant_data['var1'], label=f'Participant {participant_id}')
    plt.title(f'Time Series Plot for Participant {participant_id} - Variable 1')
    plt.xlabel('Date')
    plt.ylabel('Value of Variable 1')
    plt.legend()
    plt.show()



##################

# selecting rows by index
df.iloc[0]
df.iloc[[0,1]] # df.iloc[[0,1],1] for specific cell

# indexing colums/series
df[['id','time']]
df['id']

# df.loc() = location without index ~ can use names of columns
df.loc[0,'time']
df.loc[[0,1,2],'time'] # or df.loc[0:2,'time'] ~ slicing is inclusive of all mentioned values




ppl = {
       'first': ['corey','jane','john'],
       'last': ['smith','doe','doe'],
       'email': ['cc@gmail.com','janedoe@gmail.com','johndoe@gmail.com']
       }

for i in ppl.keys():
    print(ppl[i][0])
    
pd.DataFrame(ppl)

##################

# Step 1: Create dummy data
# Generate sample data
np.random.seed(0)
dates = pd.date_range('20230101', periods=100)
data = pd.DataFrame({
    'Person1': np.random.randn(100).cumsum(),
    'Person2': np.random.randn(100).cumsum()
}, index=dates)

# Step 2: Calculate Pearson correlation
pearson_corr = data['Person1'].corr(data['Person2'])
print(f"Pearson correlation: {pearson_corr}")

# Step 3: Calculate Spearman correlation
spearman_corr = data['Person1'].corr(data['Person2'], method='spearman')
print(f"Spearman correlation: {spearman_corr}")
