#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:57:04 2024

@author: esmeekool
"""

import pandas as pd
import numpy as np



df = pd.read_csv('/Users/esmeekool/Desktop/dataset_mood_smartphone.csv', delimiter=',')
df = df[df['variable'] != 'call']
df = df[df['variable'] != 'sms']
participants = df['id'].unique()
n=len(participants)
df[df['variable']=='sms']['value'].unique()
df['time'] = pd.to_datetime(df['time'])
df['rounded_time'] = df['time'].dt.to_period('D').dt.to_timestamp()




#Data Organizing
#participants=[ 'AS14.05']
#days_range= df['rounded_time'].unique()

figure_1= [] # empty dataframe to put each participant info into 

prediction_length=6 # change depending on how many days you want variables to reflect 
for person in participants:

    #Ordering the day and removing time stamps that have no mood value
    participant_data=df[df['id']== person]
    pivot_df= participant_data.pivot_table(index='rounded_time', columns='variable', values='value', aggfunc='mean')
    pivot_df = pivot_df.dropna(subset=['mood'])
    
    #deciding range of data to use
    days_range= pivot_df.index.unique()
    random_start=np.random.choice(days_range[0:-prediction_length])
    start_index = pivot_df.index.get_loc(random_start)
    random_start_dt = pd.Timestamp(random_start)
    
    #subsetting range of data to use
    subset_data = pivot_df.iloc[start_index : start_index + prediction_length]

    #average predictor variables and moo dvariable
    subset_pred= subset_data.iloc[:-1].drop(columns='mood').mean()  # Exclude 'mood' variable    
    subset_out = subset_data.iloc[-1:].filter(items=['mood']).mean()  # Only i
    
    #combine subsets 
    subset_pred
    pivot_df
    
    subset_pred['mood']=subset_out['mood']
    subset_final = pd.DataFrame(subset_pred).T  # Transpose to make the variables as rows

    figure_1.append(subset_final)
    
final_df = pd.concat(figure_1, ignore_index=True)
final_df.fillna(0, inplace=True)  #0 in place of Nan values --> not the final tactic!!!




# Random tree search 
#preparing the data (predicitona dn outcome variables seperate) 
mood = np.array(final_df['mood'])

variables= final_df.drop('mood', axis = 1)
variable_names = list(variables.columns) # variables names 
variables = np.array(variables) # numpy array instead of dataframe 

#training/test sets 
from sklearn.model_selection import train_test_split
'''
OMG this is kinda a usefull function
Just need to still decide if we simply wanta  crossover design or like if we can do k-fold
'''
train_variables, test_variables, train_mood, test_mood = train_test_split(variables, mood, test_size = 0.25, random_state = 42)


#Establishign a baseline 
#baseline_preds = test_variables[:, variable_names.index('average')] # The baseline predictions are the historical averages
##baseline_errors = abs(baseline_preds - test_mood) # Baseline errors, and display average baseline error
#print('Average baseline error: ', round(np.mean(baseline_errors), 2))

#Training the model
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42) # Instantiate model with 1000 decision trees
rf.fit(train_variables, train_mood); # training the model 

#make predictions
predictions = np.round(rf.predict(test_variables))
errors = abs(predictions - test_mood) #Calculate the absolute errors
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.') #mean absolute error (mae)


#performance metrics 
mape = 100 * (errors / test_mood) # Calculate mean absolute percentage error (MAPE)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

#visualizing
from sklearn import tree
import matplotlib.pyplot as plt

# single decision tree from the forest
plt.figure(figsize=(12, 8))
tree.plot_tree(rf.estimators_[0], feature_names=variable_names, filled=True)
plt.show()


# predictions vs actual value graphed 

plt.figure(figsize=(10, 6))
plt.scatter(test_mood, predictions, alpha=0.5)
plt.plot(test_mood, test_mood, color='red', linestyle='--')  # Line for zero error
plt.xlabel('Actual Mood')
plt.ylabel('Predicted Mood')
plt.title('Actual vs Predicted Mood')
plt.show()


    
