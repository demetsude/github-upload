#!/usr/bin/python

import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor

pd.options.mode.chained_assignment = None


class Predictor:

    def __init__(self, result_path):
        '''
            Class definition for the predictor
        '''
        self.scaler = MinMaxScaler()
        self.result_path = result_path

    def read_data(self, data_path):
        '''
            Read CSV file under <data_path>
            Input:
                data_path: string
            Return:
                pd.DataFrame
        '''
        data = pd.read_csv(data_path)
        return data

    def remove_missing_data(self, data):
        '''
            Remove the missing data
            Input:
                data: pd.DataFrame
            Return:
                pd.DataFrame
        '''
        # Drop the columns campaign_id and fb_campaign_id
        # Rearrange the columns
        complete_part = data[data.loc[:, 'total_conversion'].notnull()]
        missing_part = data[data.loc[:, 'total_conversion'].isnull()]
        complete_part = complete_part.drop(
            data.ix[:, 'campaign_id':'fb_campaign_id'].head(0).columns, axis=1)
        missing_part = missing_part.dropna(axis='columns')
        missing_part.columns = complete_part.columns
        data = pd.concat([complete_part, missing_part])
        return data

    def prepare_data(self, data):
        '''
            Prepare the data by handling with numerical and categorical
            variables and split the data into train and test sets
            Input:
                data: pd.DataFrame
            Return:
                X: np.ndarray containing all the features
                y: np.ndarray containing CTR values
        '''
        # Arrange interest columns as categorical data
        data.loc[:, "interest1"] = data.loc[:, "interest1"].astype(str)
        data.loc[:, "interest2"] = data.loc[:, "interest2"].astype(str)
        data.loc[:, "interest3"] = data.loc[:, "interest3"].astype(str)

        # Remove zero clicks
        data = data.loc[data.clicks != 0]

        # Create CTR
        ctr = data.apply(lambda row: row['clicks'] * 100.0 /
                         row['impressions'], axis=1)
        data.insert(loc=0, column='ctr', value=ctr)

        # Extract day as a numerical data
        data.loc[:, 'day'] = data.loc[:, 'reporting_start'].\
            map(lambda x: float(dt.datetime.strptime(x, '%d/%m/%Y').day))
        # Convert age intervals to average age as a numerical data
        data.loc[:, 'avg_age'] = data.loc[:, 'age'].\
            map(lambda x: sum(map(float, x.split('-')))/len(x.split('-')))

        # Extract the day of week as a categorical data and encode
        data.loc[:, 'week_day'] = data.loc[:, 'reporting_start'].\
            map(lambda x: dt.datetime.strptime(x, '%d/%m/%Y').strftime('%A'))
        data.loc[:, 'week_day'] = pd.Categorical(data.loc[:, 'week_day'])
        encoded_columns = pd.get_dummies(data.loc[:, 'week_day'],
                                         prefix='week_day')
        # Encode gender
        encoded_columns.loc[:, 'gender'] = \
            data.loc[:, 'gender'].map({'M': 1, 'F': -1})
        # Encode interest1
        data.loc[:, 'interest1'] = pd.Categorical(data.loc[:, 'interest1'])
        encoded_columns_interest1 = pd.get_dummies(data.loc[:, 'interest1'],
                                                   prefix='interest1')
        encoded_columns = pd.concat(
                          [encoded_columns_interest1,
                           encoded_columns], axis=1)
        encoded_columns = encoded_columns.astype(float)
        # Scale the numerical data
        columns_to_scale = ['ctr', 'avg_age', 'day']
        scaled_columns = self.scaler.fit_transform(
                          data.loc[:, columns_to_scale])
        # Concatenate the encoded categorical data and scaled numerical data
        scaled_encoded_data = np.concatenate(
                              [scaled_columns,
                               encoded_columns], axis=1)
        y = scaled_encoded_data[:, 0]
        X = scaled_encoded_data[:, 1:]

        return X, y

    def predict(self, X, y):
        '''
            Train the model for CTR prediction
            Input:
                X: np.ndarray
                y: np.ndarray
            Return:
                float
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=0)
        model = GradientBoostingRegressor(n_estimators=100,
                                          learning_rate=0.2,
                                          max_depth=5,
                                          random_state=0,
                                          loss='ls')
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        model_r2_score = r2_score(y_test, predictions)
        results = pd.DataFrame({'ctr_true_value': y_test,
                                'ctr_predicted_value': predictions})
        results.to_csv(os.path.join(self.result_path,
                       'test_predictions.csv'), index=False)

        # Plot true values vs predictions for the test set
        sns.regplot(x=y_test, y=predictions)
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title("CTR Predictions with r2 score = " +
                  str(round(model_r2_score, 2)))
        plt.savefig(os.path.join(self.result_path,
                    'test_data_true_vs_predictions.png'))
        return model_r2_score


if __name__ == '__main__':
    data_path = os.path.join('.', 'data', 'data.csv')
    result_path = os.path.join('.', 'result')
    app = Predictor(result_path)
    data = app.read_data(data_path)
    data = app.remove_missing_data(data)
    X, y = app.prepare_data(data)
    score = app.predict(X, y)
