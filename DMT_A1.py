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

# (number of) variables
variables = df['variable'].unique()
nr_variables = len(variables)

# datapoints per participant (= unequal)
df['id'].value_counts() # histogram ??











# data cleaning
''' how to check if everyone filled out all variables at each time point'''
df['time'].value_counts()

# set index as different time
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
# df.set_index('time')
# df.loc['AS.14.01]
# df.set_index('time').sort_index(inplace=True)


# filtering participants
# i.e., filtering based on variable (mood values only & compare -- etc.)
'https://www.youtube.com/watch?v=Lw2rlcxScZY&list=PL-osiE80TeTsWmV9i9c58mdDCSskIFdDS&index=4'






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





