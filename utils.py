"""
Author: Dimas Ahmad
Description: This file contains utility functions for the project.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans


def preprocess_data(df):
    df = df[df['LoanStatus'] != 'Cancelled']
    df['finished'] = df['LoanStatus'].apply(lambda x: 1 if x in ['Completed', 'Chargedoff', 'Defaulted'] else 0)
    df['risk'] = df['LoanStatus'].apply(lambda x:
                                        1 if x in ['Chargedoff', 'Defaulted']
                                        else (0 if x == 'Completed' else None))

    keep_cols = ['LoanKey', 'MemberKey', 'Term', 'BorrowerRate', 'LenderYield', 'EstimatedEffectiveYield',
                 'EstimatedLoss',
                 'EstimatedReturn', 'ProsperRating (numeric)', 'ProsperScore', 'EmploymentStatus',
                 'EmploymentStatusDuration', 'IsBorrowerHomeowner',
                 'CreditScoreRangeLower', 'CurrentDelinquencies', 'AmountDelinquent', 'BankcardUtilization',
                 'TradesNeverDelinquent (percentage)',
                 'DebtToIncomeRatio', 'IncomeVerifiable', 'StatedMonthlyIncome', 'LoanCurrentDaysDelinquent',
                 'LoanOriginalAmount', 'MonthlyLoanPayment',
                 'PercentFunded', 'Recommendations', 'InvestmentFromFriendsCount', 'InvestmentFromFriendsAmount',
                 'Investors', 'finished', 'risk']

    df = df[keep_cols]

    df.dropna(subset=['EstimatedReturn', 'EmploymentStatusDuration', 'DebtToIncomeRatio'], inplace=True)

    cat_cols = ['EmploymentStatus', 'IsBorrowerHomeowner', 'LoanKey', 'MemberKey', 'finished', 'risk',
                'IncomeVerifiable']
    num_cols = [col for col in df.columns if col not in cat_cols]

    df[num_cols] = df[num_cols].apply(lambda x: (x - x.mean()) / x.std())
    df = pd.get_dummies(df, columns=['EmploymentStatus'], drop_first=True)
    return df


def risk_prediction(df):
    # Logistic Regression to fill NaN in df['risk']
    df_train = df[df['finished'] == 1]
    df_test = df[df['finished'] == 0]
    features_col = [col for col in df.columns if col not in ['LoanKey', 'MemberKey', 'finished', 'risk']]

    x = df_train[features_col]
    y = df_train['risk']

    risk_model = LogisticRegression()
    risk_model.fit(x, y)
    accuracy = risk_model.score(x, y)
    print(f'Training acc: {accuracy:.4f}')
    y_pred = risk_model.predict(x)
    cm = confusion_matrix(y, y_pred)
    print(f'Training conf matrix: {cm}')

    x_test = df_test[features_col]
    df_test.loc[:, 'risk'] = risk_model.predict_proba(x_test)[:, 1]
    df_train.loc[:, 'risk'] = risk_model.predict_proba(x)[:, 1]

    df = pd.concat([df_train, df_test])
    return df


def discretize(df, k=9):
    # Discretize risk and yield
    kmeans = KMeans(n_clusters=k)
    risks = df['risk'].values.reshape(-1, 1)
    kmeans.fit(risks)

    # assign risk to the closest cluster center
    df.loc[:, 'cluster'] = kmeans.predict(risks)
    df['risk'] = df['cluster'].map({i: center[0] for i, center in enumerate(kmeans.cluster_centers_)})

    # Use AEFD to discretize yield (Approx. Equal Frequency Discretization)
    df['cluster'] = pd.qcut(df['LenderYield'], k, labels=False)
    # Compute the mean yield for each cluster
    df['LenderYield'] = df.groupby('cluster')['LenderYield'].transform('mean')

    df.drop(columns=['cluster'], inplace=True)
    return df


def get_data(path='./prosperLoanData.csv'):
    df = pd.read_csv(path)
    df = preprocess_data(df)
    df = risk_prediction(df)
    df = discretize(df)
    return df
