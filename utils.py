"""
Author: Dimas Ahmad
Description: This file contains utility functions for the project.
"""

import torch
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans


def preprocess_data(df):
    df = df.copy()
    df = df[df['LoanStatus'] != 'Cancelled']
    df.loc[:, 'finished'] = df.loc[:, 'LoanStatus'].apply(lambda x:
                                                          1 if x in ['Completed', 'Chargedoff', 'Defaulted']
                                                          else 0)
    df.loc[:, 'risk'] = df.loc[:, 'LoanStatus'].apply(lambda x:
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

    # Rename LoanKey into i1, i2, i3, ... and MemberKey into u1, u2, u3, ...
    unique_loan_keys = {key: f"i{idx + 1}" for idx, key in enumerate(df['LoanKey'].unique())}
    df['LoanKey'] = df['LoanKey'].map(unique_loan_keys)

    unique_member_keys = {key: f"u{idx + 1}" for idx, key in enumerate(df['MemberKey'].unique())}
    df['MemberKey'] = df['MemberKey'].map(unique_member_keys)
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


def get_data(path='./prosperLoanData.csv', k=9):
    df = pd.read_csv(path)
    df = preprocess_data(df)
    df = risk_prediction(df)
    df = discretize(df, k=k)
    print(df.shape)
    return df


def get_interaction_matrix(data):
    # Create a matrix of investor-loan interactions
    member_mapping = {member: idx for idx, member in enumerate(data['MemberKey'].unique())}
    loan_mapping = {loan: idx for idx, loan in enumerate(data['LoanKey'].unique())}
    risk_mapping = {risk: idx for idx, risk in enumerate(data['risk'].unique())}
    return_mapping = {ret: idx for idx, ret in enumerate(data['LenderYield'].unique())}

    n = len(member_mapping)
    m = len(loan_mapping)
    p = len(risk_mapping)
    q = len(return_mapping)

    R = np.zeros((n, m), dtype=np.int8)
    P = np.zeros((m, p), dtype=np.int8)
    Q = np.zeros((m, q), dtype=np.int8)

    # Fill the matrix using the mappings
    for _, row in data.iterrows():
        member_idx = member_mapping[row['MemberKey']]
        loan_idx = loan_mapping[row['LoanKey']]
        risk_idx = risk_mapping[row['risk']]
        return_idx = return_mapping[row['LenderYield']]

        R[member_idx, loan_idx] = 1
        P[loan_idx, risk_idx] = 1
        Q[loan_idx, return_idx] = 1

    return R, P, Q


def split_train_test(R, train_size=0.8):
    coo = R.tocoo()
    edges = np.vstack([coo.row, coo.col]).T
    np.random.shuffle(edges)

    split = int(len(edges) * train_size)
    train_edges = edges[:split]
    test_edges = edges[split:]
    return train_edges, test_edges


def get_sparse_matrices(data_path, k=9):
    data = get_data(data_path, k=k)
    R, P, Q = get_interaction_matrix(data)
    sR = sp.csr_matrix(R)
    sP = sp.csr_matrix(P)
    sQ = sp.csr_matrix(Q)

    return sR, sP, sQ


def build_adjacency(R, P, Q, weighted=False):
    n, m = R.shape
    p = P.shape[1]
    q = Q.shape[1]

    # Diagonal
    I_n = sp.eye(n, format='csr') if weighted else sp.csr_matrix((n, n))
    I_m = sp.eye(m, format='csr') if weighted else sp.csr_matrix((m, m))
    I_p = sp.eye(p, format='csr') if weighted else sp.csr_matrix((p, p))
    I_q = sp.eye(q, format='csr') if weighted else sp.csr_matrix((q, q))

    # Zero matrices for padding
    Z_np = sp.csr_matrix((n, p))
    Z_nq = sp.csr_matrix((n, q))
    Z_qp = sp.csr_matrix((q, p))
    Z_pq = sp.csr_matrix((p, q))

    top_row = sp.hstack([I_n, R, Z_np, Z_nq], format='csr')
    second_row = sp.hstack([R.T, I_m, P, Q], format='csr')
    third_row = sp.hstack([Z_np.T, P.T, I_p, Z_pq], format='csr')
    fourth_row = sp.hstack([Z_nq.T, Q.T, Z_qp, I_q], format='csr')

    A = sp.vstack([top_row, second_row, third_row, fourth_row], format='csr')
    return A


def symmetric_normalization(A):
    degree = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(degree, -0.5, out=np.zeros_like(degree, dtype=np.float32), where=degree > 0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


def to_torch_sparse(A):
    coo = A.tocoo()
    indices = torch.stack([torch.LongTensor(coo.row), torch.LongTensor(coo.col)])
    values = torch.FloatTensor(coo.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape)).coalesce()
