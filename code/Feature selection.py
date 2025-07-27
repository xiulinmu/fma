#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Recursive feature elimination
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skopt import BayesSearchCV

data = pd.read_csv('Candidatus Brocadia original data.csv')
X = data.drop('Candidatus Brocadia', axis=1)
y = data['Candidatus Brocadia']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

param_space = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 1, 5]
}

results = []


for i in range(2, X.shape[1]+1):
    print(f"\n==== Current number of retained features: {i} ====")

    base_model = XGBRegressor(random_state=42)
    selector = RFE(base_model, n_features_to_select=i, step=1)
    selector = selector.fit(X, y)
    selected_features = X.columns[selector.support_]
    X_selected = X[selected_features]

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    opt = BayesSearchCV(
        estimator=XGBRegressor(random_state=42),
        search_spaces=param_space,
        n_iter=80,
        cv=5,
        n_jobs=-1,
        random_state=42
    )
    opt.fit(X_train, y_train)

    best_model = opt.best_estimator_
    best_model.fit(X_train, y_train)

    y_train_pred = best_model.predict(X_train)

    y_test_pred = best_model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("Select features：", list(selected_features))
    print("Train RMSE: {:.3f}, MAE: {:.3f}, R²: {:.3f}".format(train_rmse, train_mae, train_r2))
    print("Test RMSE: {:.3f}, MAE: {:.3f}, R²: {:.3f}".format(test_rmse, test_mae, test_r2))
    
#Mutual information
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.wrappers.scikit_learn import KerasRegressor
from skopt import BayesSearchCV
from skopt import space


data = pd.read_csv('NRR original data.csv')
X = data.drop('NRR', axis=1)
y = data['NRR']


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)


mi = mutual_info_regression(X, y)
mi_scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)

param_space = {
    'units1': space.Integer(32, 256),
    'learning_rate': space.Real(1e-4, 1e-1, prior='log-uniform'),
    'activation': space.Categorical(['relu', 'tanh', 'sigmoid']),
    'optimizer': space.Categorical(['adam', 'sgd', 'rmsprop']),
    'dropout_rate': space.Real(0.1, 0.5)
}

def create_model(units1, learning_rate, activation, optimizer, dropout_rate):
    model = Sequential()
    model.add(Dense(units1, input_dim=input_dim, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))

    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)

    model.compile(optimizer=opt, loss='mean_squared_error')
    return model

results = []

for i in range(5, 15):  
    top_features = mi_scores.index[:i]
    X_selected = X[top_features]

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    input_dim = X_selected.shape[1]
    keras_model = KerasRegressor(build_fn=create_model, verbose=0)

    opt = BayesSearchCV(
        estimator=keras_model,
        search_spaces=param_space,
        n_iter=80,
        cv=5,
        n_jobs=1,
        random_state=42
    )

    opt.fit(X_train, y_train)

    best_model = create_model(**opt.best_params_)
    best_model.fit(X_train, y_train, epochs=100, batch_size=50, verbose=0)

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    results.append({
        'features_used': i,
        'selected_features': list(top_features),
        'Train_RMSE': train_rmse,
        'Train_MAE': train_mae,
        'Train_R2': train_r2,
        'Test_RMSE': test_rmse,
        'Test_MAE': test_mae,
        'Test_R2': test_r2
    })


results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv('Select features and performance.csv', index=False)

