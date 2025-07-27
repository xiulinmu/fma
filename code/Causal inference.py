#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Causal inference
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('NRR original data.csv')
print("Columns in dataset:", df.columns)

Y = df['FA'].values                          

feature_columns = [col for col in df.columns if col != 'FA']

kf = KFold(n_splits=10, shuffle=True, random_state=42)

for treatment_col in feature_columns:
    if treatment_col == 'FA':
        continue

    print(f"\n=== Processing treatment variable: {treatment_col} ===")
    
    T = df[treatment_col].values.reshape(-1, 1)   
    X = df.drop(columns=[treatment_col, 'FA'])   

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ate_list = []
    pvalue_list = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        T_train, T_val = T[train_idx], T[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        model_y = RandomForestRegressor(n_estimators=200, random_state=0)
        model_t = RandomForestRegressor(n_estimators=200, random_state=0)

        est = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=False,
            random_state=0
        )

        est.fit(Y_train, T_train, X=X_train)

        ate_point = est.ate(X_val)
        ate_infer = est.ate_inference(X_val)

        ate_list.append(ate_point)

    print(f"--> Mean ATE for {treatment_col}: {np.mean(ate_list):.2f}")

