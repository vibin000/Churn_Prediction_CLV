
# -*- coding: utf-8 -*-

#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib
import pandas as pd
pd.options.mode.chained_assignment = None 

#%%

#LOADING THE DATASET
#MEISTER

licenses = pd.read_csv('Please provide path/licenses_paid.csv',sep = ',')
#Product registrations
activity = pd.read_csv('Please provide path/Meister/mt_activity.csv',sep=',')


#%%

#Feature Engineering ,variable transformation and creation ,and data binning

class dataPreparation:
    """Takes in the licenses dataset and activity dataset and clean the datasets based on non_null team_ids as they are the
    connection feature between the two datasets.The date features will get tarnsformed into timestamp and recency
    variable for modelling. The plans feature column in the licesnses dataset will be binned into a smaller subset and finally will be one-hot encoded
    into dummy variables.Finally the dataset will be aggregated on team_id taking "sum"" as aggregation function.The activity dataset will also be 
    worked upon to create an aggregated final dataset with both monthly and total activity aggregated based on taem_id"""
    
    def __init__(self,dataset1,dataset2):
        self.dataset_license = dataset1
        self.dataset_activity = dataset2
        
        
    def data_preparation_licenses(self): 
        """Function that prepared the licenses dataset- Feature Engineering such a null value filtering,variable
        binning and aggreagation is done here"""
        licenses = self.dataset_license
        #Filtering out the non null values
        licenses= licenses[licenses.team_id.notnull()]

        #Datetime transformation and recency variable creation
        #licenses['license_end'] = licenses['license_end'].str[:6]+'20'+ licenses['license_end'].str[6:]
        licenses['license_end'] = pd.to_datetime(licenses['license_end'], format='%Y-%m-%d')
        licenses['license_end_recency'] = licenses.apply(lambda x : (dt.today()-x.license_end).days ,axis = 1) 
        licenses['license_start'] = pd.to_datetime(licenses['license_start'], format='%Y-%m-%d')
        licenses['license_start_recency'] = licenses.apply(lambda x : (dt.today()-x.license_start).days ,axis = 1) 
        
        
        #For simplicity ,for now ,merging all the pro plans as pro ,acedemic as academic and so on--DATA CATEGORICAL BINNING
        replace_values = {'pro1':'pro', 'business12':'business', 'pro12':'pro', 'business1':'business', 'business6':'business', 'edu12':'edu',
                           'business36':'business', 'business24':'business', 'pro6':'pro', 'edu6':'edu', 'business18':'business',
                           'academic1':'academic', 'enterprise12':'enterprise', 'pro36':'pro', 'edu36':'edu', 'academic12':'academic',
                           'pro18':'pro', 'business3':'business', 'pro24':'pro', 'edu24':'edu', 'enterprise36':'enterprise', 'pro3':'pro',
                           'edu3':'edu', 'edu60':'edu'}
        
        licenses.replace({'plan':replace_values},inplace=True)
        
        #Creating dummy variable to factor in the information about the plan for predicting the churn
        licenses = pd.get_dummies(licenses,columns=['plan'])
        return licenses 

    def data_preparation_activity(self):
        """Function that prepares the activity dataset.Dataset filtering based on null value of Team_ID is done.
        The dataset is also filted based on available team_ids from lisences dataset as they are the key connectors in our case.
        Aggregation based on both monthly data and total data are made always on team_id."""
        
        activity = self.dataset_activity
        
        #Datetime transformation and creation of year ,month variable for EDA
        #activity['event_date'] = activity['event_date'].str[:6]+'20'+ activity['event_date'].str[6:]
        activity['event_date'] =  pd.to_datetime(activity['event_date'], format='%Y-%m-%d')
        activity['year'] = activity['event_date'].dt.year
        activity['month'] = activity['event_date'].dt.month
        
        #Filtering oput non nan values
        activiy = activity[activity.team_id.notnull()]
        
        
        #taking unique team ids from license dataset as they are the connection key        
        licenses_unique = self.dataset_license[self.dataset_license.team_id.notnull()].team_id.unique().tolist()
        #licenses_unique = [int(x) for x in licenses_unique]
        
        
        #Filtering out only the paid team_ids since the churn is measured only for paid customers
        activity_paid = activity[activity.team_id.isin(licenses_unique)]
        #Filling the missing value with 0 taking into account the logic the nan values are actually the non-usage of a specific activity
        activity_paid.fillna(0,inplace=True)
        
        
        #Creating a monthly average usage for each team_id
        activity_monthly = activity_paid.groupby(['team_id','month','year']).agg({'user_id':'nunique','projects_created':'sum',
                                                                                  'tasks_created':'sum', 'checklists_created':'sum', 'agenda_events':'sum',
                                                                                  'timeline_events':'sum', 'reports_opened':'sum', 'reports_saved':'sum',
                                                                                  'invitations_sent':'sum', 'invitations_claimed':'sum',
                                                                                  'project_groups_created':'sum'}).reset_index()
        
        activity_monthly = activity_monthly.groupby(['team_id']).agg({'user_id':'mean','projects_created':'mean',
                                                                                  'tasks_created':'mean', 'checklists_created':'mean', 'agenda_events':'mean',
                                                                                  'timeline_events':'mean', 'reports_opened':'mean', 'reports_saved':'mean',
                                                                                  'invitations_sent':'mean', 'invitations_claimed':'mean',
                                                                                  'project_groups_created':'mean'}).reset_index()
        
        aggregated_results_monthly = activity_monthly.mean(axis=0)
        
        #Renaming columns for easy understanding
        activity_monthly.columns= ['team_id', 'user_id_monthly', 'projects_created_monthly', 'tasks_created_monthly',
               'checklists_created_monthly', 'agenda_events_monthly', 'timeline_events_monthly',
               'reports_opened_monthly', 'reports_saved_monthly', 'invitations_sent_monthly',
               'invitations_claimed_monthly', 'project_groups_created_monthly']
        
        #Creating total usage of each team
        activity_total = activity_paid.groupby(['team_id']).agg({'user_id':'nunique','projects_created':'sum',
                                                                                  'tasks_created':'sum', 'checklists_created':'sum', 'agenda_events':'sum',
                                                                                  'timeline_events':'sum', 'reports_opened':'sum', 'reports_saved':'sum',
                                                                                  'invitations_sent':'sum', 'invitations_claimed':'sum',
                                                                                  'project_groups_created':'sum'}).reset_index()

        #Merging both
        activity_total = pd.merge(activity_monthly,activity_total,on='team_id',how='left')
        
        
        return activity_total


    def clv_avg_repurchase(self):
        """This fucntion will output the days for avaerage repurchases which is then to be used for target variable creation"""
        licenses = self.data_preparation_licenses()
        #Grouping by recency and team id to  filter out the same day purchases
        licenses_grouped = licenses.groupby(['team_id','license_start_recency']).agg({'quantity':'sum'}).reset_index()
        
        
        #ordering based on the first recency just to calculate the mean time difference for each team in their repurchase 
        licenses_grouped.sort_values(['team_id','license_start_recency'],ascending=[False,True],inplace=True)
        licenses_grouped = licenses_grouped.reset_index()
        
        
        #Adding a cumilative count to fitler out only the team ids with more than 1 purchase
        licenses_grouped['cum_seq'] = licenses_grouped.groupby(['team_id']).cumcount()+1
        #Taking only the ids which have atleast two purchases for calculating Customer Life cycle
        #Customer Life Cycle
        #FIltering out only the lisence ids with more than 1 purchase
        licences_clv = licenses_grouped[licenses_grouped.team_id.isin(licenses_grouped[licenses_grouped.cum_seq==2].team_id.unique().tolist())].reset_index()
        licences_clv.drop(columns=['level_0','index'],inplace=True)
        
        
        #Aggregating w.r.t team_id taking the sum of days differences between each purchase for each team id.This will give us the total days between the first and last purchase
        clv = licences_clv.groupby('team_id').apply(lambda x: ((x['license_start_recency']-x['license_start_recency'].shift()).fillna(0)).sum()).reset_index()
        #Calculating the number of purchases of each tem id to divide it with the 
        clv_number = licences_clv.groupby(['team_id']).agg({'cum_seq':'max'}).reset_index()
        
        clv = pd.merge(clv,clv_number,how='left',on='team_id')
        #Subtracting by 1 as the total purchase will be 1 row less for each team id.
        clv['cum_seq'] = clv['cum_seq']-1
        #Creating an average purchase column for each team id
        clv['avg_repurchase_days'] = clv[0]/clv['cum_seq']
        
        #Average repurchase recency
        #We are taking the outliers that are greater than 365 days and less than 30days as the dataset is only for 1.5 years
        
        avg_repurchase = clv[(clv.avg_repurchase_days <=365) & (clv.avg_repurchase_days >=30)].avg_repurchase_days.mean()

        return avg_repurchase
    
            
    def prediction_dataset(self):
        """This function will create the final dataset for prediction with the cleaned data and the two target variables"""
        #Calling the prepared licesnces dataset,activity dataset and the average repurchase days found from CLV analysis.
        licenses = self.data_preparation_licenses()
        avg_repurchase = self.clv_avg_repurchase()
        activity_total = self.data_preparation_activity()
        
        #Licenses dataset with the categorical variables aggregated based on their frequency
        licenses_final = licenses.groupby(['team_id']).agg({'plan_academic':'sum', 'plan_business':'sum', 'plan_edu':'sum',
                                                            'plan_enterprise':'sum', 'plan_pro':'sum', 'license_end_recency':'sum',
                                                            'quantity':'sum'}).reset_index()

        
        #Creating an aggregated dataset of liceses data for the target variable creation
        target_variable = licenses.groupby(['team_id']).agg({'license_start_recency':'min','license_end_recency':'min'}).reset_index()

        #Target 1 -- Logic ,taking an assumption that the customers who have not bought a product 165 days(average repurchase time) after their last lisence end date
        target_variable['Target1'] = np.where((target_variable['license_end_recency']>avg_repurchase),1,0)
        #Target 2 -- Logic ,taking an assumption that the customers who have not bought a product 65 days(average repurchase time) after their last purchase date and their lisence have already ended 
        target_variable['Target2'] = np.where((target_variable['license_start_recency']>avg_repurchase)& (target_variable['license_end_recency']>0),1,0)

        #Final prediction dataset with targe variable
        prediction_dataset = pd.merge(licenses_final,target_variable,on='team_id',how='left')
        
        #also merging the activity dataset for predictive variables
        prediction_dataset = pd.merge(activity_total, prediction_dataset,on='team_id',how='left')

        return prediction_dataset



class churnModel:
    def __init__(self,dataset_prediction):
        self.prediction_dataset = dataset_prediction
        
    def model_pipelining(self):
        """This fuction will give us the best hyperparameter for the random forrest/or another training model
        we are going train as output.Note this is just for TARGET 1"""
        prediction_dataset = self.prediction_dataset
        #For the feature columns we remove the team_id ,the target variable and recency as itwill lead to data leakage
        feature_columns = [x for x in prediction_dataset.columns.tolist() if x not in ['team_id','license_start_recency','license_end_recency_y','license_end_recency_x', 'Target1','Target2']]
        #Taking 75% are training and 25% as testing datset
        X_train, X_test, y_train, y_test = train_test_split(prediction_dataset[feature_columns], 
                                                            prediction_dataset.Target1,
                                                            test_size=0.25 , random_state=0)
          
        #Model 1 -RANDOM FOREST CLASSIFIER with target 1
        #Random forrest
        
        #Creating a pipeline with multiple hyper parameters to train the model
        pipe = Pipeline([
                        ("clf", RandomForestClassifier())
                        ])
        
        param_grid = {"clf__n_estimators": [100, 500, 1000],
                      "clf__max_depth": [5,10,25],#, 5, 10, 25],
                      "clf__max_features": [*np.arange(0.1, 1.1, 0.1)]
                      }
        
        #Initiating a grid search ,with 3 fold cross validation to search through each combination of parameter assigned in the pipeline
        
        gs = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=4, verbose=2)
        gs.fit(X_train, y_train)
        
        best_params = gs.best_params_
        #Best parameters are 'clf__max_depth': 10, 'clf__max_features': 0.9, 'clf__n_estimators': 500

        #We can train multiple models with the same pipelining logic.
        #Here for simplicity we only train one model and the CONFUSION MATRIX ,accuracy precision etc will be checked later 
    
        return best_params

    def random_forest(self):
        """This fucntion will train the datset with the best params found and outputs the model for prediction"""
        prediction_dataset = self.prediction_dataset
        feature_columns = [x for x in prediction_dataset.columns.tolist() if x not in ['team_id','license_start_recency','license_end_recency_y','license_end_recency_x', 'Target1','Target2']]
        X_train, X_test, y_train, y_test = train_test_split(prediction_dataset[feature_columns], 
                                                            prediction_dataset.Target1,
                                                            test_size=0.25 , random_state=0)

        ###Now we can use the best hyperparameters found using the gridsearch cv to build a new model
        ###With the new model trained ,the accuracy can be tested using the confusion matrix and the feature importances and other testing can be done
        #Example of model
        param = self.model_pipelining()
        number_estimators =param['clf__n_estimators']
        depth = param['clf__max_depth']
        features = param['clf__max_features']
        rf_model = RandomForestClassifier(n_estimators=number_estimators,max_depth=depth,max_features=features)
        rf_model.fit(X_train, y_train)


        return rf_model




