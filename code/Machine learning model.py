#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#ANN
import shap
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from skopt import space
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from scipy.stats import gaussian_kde
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasRegressor
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
data = pd.read_csv('NRR feature selection.csv')
X = data.drop('NRR', axis=1)
y = data['NRR']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

param_space = {
    'units1': space.Integer(32, 256),
    'learning_rate': space.Real(1e-4, 1e-1, prior='log-uniform'),
    'activation': space.Categorical(['relu', 'tanh', 'sigmoid']),
    'optimizer': space.Categorical(['adam', 'sgd', 'rmsprop']),
    'dropout_rate': space.Real(0.1, 0.5)
}

def create_model(units1, learning_rate, activation, optimizer, dropout_rate):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units1, input_dim=14, activation=activation),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    model.compile(optimizer=opt, loss='mean_squared_error')
    return model


keras_model = KerasRegressor(build_fn=create_model)

opt = BayesSearchCV(
    keras_model,
    param_space,
    n_iter=80,
    cv=5,
    n_jobs=1
)

opt.fit(X_train, y_train)

best_params = opt.best_params_
print("Best Hyperparameters:", best_params)

results_df = pd.DataFrame(opt.cv_results_)

results_df['score'] = -results_df['mean_test_score']  

plt.figure(figsize=(7, 5), dpi=300)
plt.plot(results_df['score'], marker='o', linewidth=2)
plt.xlabel('Iteration', fontsize=16, family='Times New Roman')
plt.ylabel('CV RMSE', fontsize=16, family='Times New Roman')
plt.title('Bayesian Optimization Convergence', fontsize=18, family='Times New Roman')
plt.grid(True)
plt.xticks(fontsize=14, fontname='Times New Roman')
plt.yticks(fontsize=14, fontname='Times New Roman')
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 3), dpi=300)
plt.scatter(results_df['param_learning_rate'], results_df['score'], alpha=0.7, color='blue')
plt.xlabel('Learning Rate', fontsize=14, family='Times New Roman')
plt.ylabel('RMSE', fontsize=14, family='Times New Roman')
plt.title('Learning Rate vs RMSE', fontsize=16, family='Times New Roman')
plt.grid(True)
plt.xscale('log')
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 3), dpi=300)
plt.scatter(results_df['param_units1'], results_df['score'], alpha=0.7, color='green')
plt.xlabel('Neuron nodes', fontsize=14, family='Times New Roman')
plt.ylabel('RMSE', fontsize=14, family='Times New Roman')
plt.title('Neuron nodes vs RMSE', fontsize=16, family='Times New Roman')
plt.grid(True)
plt.tight_layout()
plt.show()

best_model = create_model(**best_params)
history = best_model.fit(X_train, y_train, epochs=100, batch_size=50, validation_data=(X_test, y_test), verbose=1)

train_rmse = np.sqrt(np.array(history.history['loss']))  
val_rmse = np.sqrt(np.array(history.history['val_loss']))  

plt.figure(figsize=(7, 5), dpi=300)
plt.plot(train_rmse, label='Train RMSE', linewidth=2)
plt.plot(val_rmse, label='Test RMSE', linewidth=2)
plt.xlabel('Epochs', fontsize=16, family='Times New Roman')
plt.ylabel('RMSE', fontsize=16, family='Times New Roman')
plt.title('Train vs Test RMSE Curve', fontsize=18, family='Times New Roman')
plt.legend(fontsize=14)
plt.grid(True)
plt.xticks(fontsize=14, fontname='Times New Roman')
plt.yticks(fontsize=14, fontname='Times New Roman')
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5), dpi=300)
plt.plot(history.history['loss'], label='Train Loss (MSE)', linewidth=2)
plt.plot(history.history['val_loss'], label='Test Loss (MSE)', linewidth=2)
plt.xlabel('Epochs', fontsize=16, family='Times New Roman')
plt.ylabel('Loss (MSE)', fontsize=16, family='Times New Roman')
plt.title('Train vs Test Loss Curve', fontsize=18, family='Times New Roman')
plt.legend(fontsize=14)
plt.grid(True)
plt.xticks(fontsize=14, fontname='Times New Roman')
plt.yticks(fontsize=14, fontname='Times New Roman')
plt.tight_layout()
plt.show()

test_loss = best_model.evaluate(X_test, y_test)

y_train_pred = best_model.predict(X_train).ravel()
y_test_pred = best_model.predict(X_test).ravel()

train_rmse_final = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse_final = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train RMSE: {train_rmse_final:.2f}, Test RMSE: {test_rmse_final:.2f}")
print(f"Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")
print(f"Train R²: {train_r2:.2f}, Test R²: {test_r2:.2f}")

residuals = y_test - y_test_pred

plt.figure(figsize=(6, 4), dpi=300)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Quantile-Quantile Plot of Residuals", fontsize=16, family='Times New Roman')
plt.xlabel("Theoretical Quantiles", fontsize=14, family='Times New Roman')
plt.ylabel("Sample Quantiles", fontsize=14, family='Times New Roman')
plt.grid(True)
plt.xticks(fontsize=12, fontname='Times New Roman')
plt.yticks(fontsize=12, fontname='Times New Roman')
plt.tight_layout()
plt.show()
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'

xy = np.vstack([y_train, y_train_pred])
z = gaussian_kde(xy)(xy)  
xy1 = np.vstack([y_test, y_test_pred])
z1 = gaussian_kde(xy1)(xy1) 
data_range = [int(min(y_train)), int(max(y_train))]
def get_regression_line(y_train, y_train_pred, data_range=(0, 1000)):
    def slope(xs, ys):
        m = (((np.mean(xs) * np.mean(ys)) - np.mean(xs * ys)) / ((np.mean(xs) * np.mean(xs)) - np.mean(xs * xs)))
        b = np.mean(ys) - m * np.mean(xs)
        return m, b
    k, b = slope(y_train, y_train_pred)
    regression_line = []
    for a in range(data_range[0], data_range[1]+1):
        regression_line.append((k * a) + b)
    return regression_line
colors = ['#FFB6C1', '#FF1493']  

start_color = (240/255, 148/255, 150/255)  
end_color = (231/255, 60/255, 54/255)        
custom_cmap = LinearSegmentedColormap.from_list('custom', [start_color, end_color], N=256)
plt.figure(figsize=(7, 5), dpi=300)  
offset = 0.2 * np.max(y_train)
plt.plot([-0.5, 5], [-0.5, 5], color='gray', linestyle='-', lw=1.5)
RMSE=0.8*train_rmse_final+0.2*test_rmse_final
plt.plot(np.arange(-0.05, 2.19), np.arange(-0.05, 2.19) + RMSE, color='gray', linestyle='--', linewidth=1.5, label='20% Offset')
plt.plot(np.arange(-0.05, 2.19), np.arange(-0.05, 2.19) - RMSE, color='gray', linestyle='--', linewidth=1.5)
plt.scatter(y_train, y_train_pred, c=z*200, cmap=custom_cmap, alpha=0.5, s=400, marker='o', edgecolors='none', label='y_test_flat1')
plt.scatter(y_test, y_test_pred, c=z1*200, cmap=custom_cmap, alpha=0.5, s=400, marker='^', edgecolors='none', label='y_test_flat')
plt.xlabel('Actual NRR', family='Times New Roman', fontsize=18)
plt.ylabel('Predicted NRR', family='Times New Roman', fontsize=18)
plt.yticks(fontproperties='Times New Roman', fontsize=18)

cbar = plt.colorbar(pad=0.04)

cbar.ax.tick_params(labelsize=14)
plt.text(0.05, 1, '$R^2=%.2f$' % test_r2, family='Times New Roman', fontsize=17)
plt.text(0.05, 0.9, '$MAE=%.2f$' % test_mae, family='Times New Roman', fontsize=17)
plt.text(0.05, 0.8, '$RMSE=%.2f$' % test_rmse_final, family='Times New Roman', fontsize=17)
plt.xlim(-0.05, 1.119)  
plt.ylim(-0.05, 1.119)  

plt.xticks(fontname='Times New Roman', fontsize=18)
plt.yticks(fontname='Times New Roman', fontsize=18)
plt.plot(0.8, 0.2, 'o', color=(231/255, 60/255, 54/255), markersize=16)
plt.text(0.9, 0.2, "train", fontsize=18, color='black', va='center', family = 'Times New Roman')
plt.plot(0.8, 0.1, '^', color=(231/255, 60/255, 54/255), markersize=16)
plt.text(0.9, 0.1, "test", fontsize=18, color='black', va='center', family = 'Times New Roman')
plt.show()

best_model_cv = KerasRegressor(build_fn=create_model, **best_params, epochs=100, batch_size=50, verbose=0)

cv_r2 = cross_val_score(best_model_cv, X_train, y_train, cv=kf, scoring='r2')

cv_rmse = cross_val_score(best_model_cv, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')

cv_mae = cross_val_score(best_model_cv, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')

print("📊 10-Fold Cross-Validation Results:")
print("R² (mean ± SD): {:.3f} ± {:.3f}".format(np.mean(cv_r2), np.std(cv_r2)))
print("RMSE (mean ± SD): {:.3f} ± {:.3f}".format(-np.mean(cv_rmse), np.std(cv_rmse)))
print("MAE (mean ± SD): {:.3f} ± {:.3f}".format(-np.mean(cv_mae), np.std(cv_mae)))


#XGboost CB 

import csv
import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import randint
from skopt import BayesSearchCV
from xgboost import XGBRegressor
from skopt.space import Real, Integer
from scipy.stats import gaussian_kde
from sklearn.model_selection import KFold
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


data = pd.read_csv('NOB feature selection.csv')
X = data.drop('NOB', axis=1)
y = data['NOB']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

kf = KFold(n_splits=10, shuffle=True, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]


param_space = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 1, 5]
}

def create_xgb_model(n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma):
    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
       gamma=gamma
    )

xgb_model = XGBRegressor()


opt = BayesSearchCV(
    xgb_model,
    param_space,
    n_iter=80,
    cv=5,
    n_jobs=1
)
opt.fit(X_train, y_train)
best_params = opt.best_params_
print("Best Parameters:", best_params)
results_df = pd.DataFrame(opt.cv_results_)
results_df['score'] = -results_df['mean_test_score'] 

plt.figure(figsize=(7, 5), dpi=300)
plt.plot(results_df['score'], marker='o', linewidth=2)
plt.xlabel('Iteration', fontsize=16, family='Times New Roman')
plt.ylabel('CV RMSE', fontsize=16, family='Times New Roman')
plt.title('Bayesian Optimization Convergence', fontsize=18, family='Times New Roman')
plt.grid(True)
plt.xticks(fontsize=14, fontname='Times New Roman')
plt.yticks(fontsize=14, fontname='Times New Roman')
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 3), dpi=300)
plt.scatter(results_df['param_learning_rate'], results_df['score'], alpha=0.7, color='blue')
plt.xlabel('Learning Rate', fontsize=14, family='Times New Roman')
plt.ylabel('RMSE', fontsize=14, family='Times New Roman')
plt.title('Learning Rate vs RMSE', fontsize=16, family='Times New Roman')
plt.grid(True)
plt.xscale('log')
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 3), dpi=300)
plt.scatter(results_df['param_max_depth'], results_df['score'], alpha=0.7, color='green')
plt.xlabel('Max Depth', fontsize=14, family='Times New Roman')
plt.ylabel('RMSE', fontsize=14, family='Times New Roman')
plt.title('Max Depth vs RMSE', fontsize=16, family='Times New Roman')
plt.grid(True)
plt.tight_layout()
plt.show()

model = XGBRegressor(**best_params)
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train,
          eval_set=eval_set,
          eval_metric='rmse',
          early_stopping_rounds=10,
          verbose=False)
results = model.evals_result()

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
print("Train RMSE: {:.3f}, MAE: {:.3f}, R²: {:.3f}".format(train_rmse, train_mae, train_r2))
print("Test RMSE: {:.3f}, MAE: {:.3f}, R²: {:.3f}".format(test_rmse, test_mae, test_r2))

plt.figure(figsize=(7, 5), dpi=300)
plt.plot(results['validation_0']['rmse'], label='Train RMSE', linewidth=2)
plt.plot(results['validation_1']['rmse'], label='Test RMSE', linewidth=2)
plt.xlabel('Boosting Iteration', fontsize=16, family='Times New Roman')
plt.ylabel('RMSE', fontsize=16, family='Times New Roman')
plt.title('Train vs Test RMSE Curve', fontsize=18, family='Times New Roman')
plt.legend(fontsize=14)
plt.grid(True)
plt.xticks(fontsize=14, fontname='Times New Roman')
plt.yticks(fontsize=14, fontname='Times New Roman')
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5), dpi=300)
plt.plot(results['validation_0']['rmse'], label='Train RMSE', linewidth=2)
plt.plot(results['validation_1']['rmse'], label='Test RMSE', linewidth=2)
plt.xlabel('Boosting Rounds', fontsize=16, family='Times New Roman')
plt.ylabel('Loss (RMSE)', fontsize=16, family='Times New Roman')
plt.title('Train and Test Loss Curves', fontsize=18, family='Times New Roman')
plt.legend(fontsize=14)
plt.grid(True)
plt.xticks(fontsize=14, fontname='Times New Roman')
plt.yticks(fontsize=14, fontname='Times New Roman')
plt.tight_layout()
plt.show()

xy = np.vstack([y_train, y_train_pred])
z = gaussian_kde(xy)(xy)  
xy1 = np.vstack([y_test, y_test_pred])
z1 = gaussian_kde(xy1)(xy1) 
data_range = [int(min(y_train)), int(max(y_train))]
def get_regression_line(y_train, y_train_pred, data_range=(0, 1000)):
    def slope(xs, ys):
        m = (((np.mean(xs) * np.mean(ys)) - np.mean(xs * ys)) / ((np.mean(xs) * np.mean(xs)) - np.mean(xs * xs)))
        b = np.mean(ys) - m * np.mean(xs)
        return m, b
    k, b = slope(y_train, y_train_pred)
    regression_line = []
    for a in range(data_range[0], data_range[1]+1):
        regression_line.append((k * a) + b)
    return regression_line

start_color = (179/255, 219/255, 97/255)  
end_color = (44/255, 115/255, 57/255)   
custom_cmap = LinearSegmentedColormap.from_list('custom', [start_color, end_color], N=256)
plt.figure(figsize=(7, 5), dpi=300)  
offset = 0.2 * np.max(y_train)

RMSE=0.8*train_rmse+0.2*test_rmse
plt.plot([-1, 39], [-1, 39], color='gray', linestyle='-', lw=1.5)
plt.plot(np.arange(-1, 40), np.arange(-1, 40) + RMSE, color='gray', linestyle='--', linewidth=1.5, label='20% Offset')
plt.plot(np.arange(-1, 40), np.arange(-1, 40) - RMSE, color='gray', linestyle='--', linewidth=1.5)
plt.scatter(y_train, y_train_pred, c=z*200, cmap=custom_cmap, alpha=0.5, s=400, marker='o', edgecolors='none', label='y_test_flat1')
plt.scatter(y_test, y_test_pred, c=z1*200, cmap=custom_cmap, alpha=0.5, s=400, marker='^', edgecolors='none', label='y_test_flat')
plt.xlabel('Actual NOB', family='Times New Roman', fontsize=20)
plt.ylabel('Predicted NOB', family='Times New Roman', fontsize=20)

cbar = plt.colorbar(pad=0.04)

cbar.ax.tick_params(labelsize=15)

plt.text(2,34, '$R^2=%.2f$' % test_r2, family = 'Times New Roman', fontsize=18)
plt.text(2,31, '$MAE=%.2f$' % test_mae, family = 'Times New Roman', fontsize=18)
plt.text(2,28, '$RMSE=%.2f$' % test_rmse, family = 'Times New Roman', fontsize=18)
plt.xlim(-1, 39)  
plt.ylim(-1, 39)  

plt.xticks(fontname='Times New Roman', fontsize=18)
plt.yticks(fontname='Times New Roman', fontsize=18)
plt.plot(30, 6, 'o', color=(44/255, 115/255, 57/255), markersize=17)
plt.text(33, 6, "train", fontsize=18, color='black', va='center', family = 'Times New Roman')
plt.plot(30, 3, '^', color=(44/255, 115/255, 57/255), markersize=17)
plt.text(33, 3, "test", fontsize=18, color='black', va='center', family = 'Times New Roman')
plt.show()

residuals = y_test - y_test_pred


plt.figure(figsize=(6, 4), dpi=300)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ Plot of Residuals ", fontsize=16, family='Times New Roman')
plt.xlabel("Theoretical Quantiles", fontsize=14, family='Times New Roman')
plt.ylabel("Sample Quantiles", fontsize=14, family='Times New Roman')
plt.grid(True)
plt.xticks(fontsize=12, fontname='Times New Roman')
plt.yticks(fontsize=12, fontname='Times New Roman')
plt.tight_layout()
plt.show()

#RF
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import randint
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from scipy.stats import gaussian_kde
from sklearn.model_selection import KFold
from sklearn.inspection import partial_dependence
from pdpbox import pdp, get_dataset, info_plots
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv('NOB original data.csv')
X = data.drop('NOB', axis=1)
y = data['NOB']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(X.columns)


n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
rf_model = RandomForestRegressor()    

param_space = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

def create_rf_model(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features
    )

opt = BayesSearchCV(
    rf_model,
    param_space,
    n_iter=80,  
    cv=5,  
    n_jobs=-1  
)

opt.fit(X_train, y_train)

best_params = opt.best_params_
print("Best Hyperparameters:", best_params)

best_rf_model = create_rf_model(**best_params)
best_rf_model.fit(X_train, y_train)


y_pred1 = best_rf_model.predict(X_train).ravel()

y_pred = best_rf_model.predict(X_test)

y_test_flat1 = np.ravel(y_train)
y_pred_flat1 = np.ravel(y_pred1)
y_test_flat = np.ravel(y_test)
y_pred_flat = np.ravel(y_pred)

rmse1 = np.sqrt(mean_squared_error(y_train, y_pred1))
print("Train RMSE: {:.3f}".format(rmse1))
mae1 = mean_absolute_error(y_train, y_pred1)
print("Train MAE: {:.3f}".format(mae1))
r21 = r2_score(y_train, y_pred1)
print("Train R²: {:.3f}".format(r21))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Test RMSE: {:.3f}".format(rmse))
mae = mean_absolute_error(y_test, y_pred)
print("Test MAE: {:.3f}".format(mae))
r2 = r2_score(y_test, y_pred)
print("Test R²: {:.3f}".format(r2))

#SVR
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from scipy.stats import randint
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from scipy.stats import gaussian_kde
from sklearn.model_selection import KFold
from pdpbox import pdp, get_dataset, info_plots
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import partial_dependence
from sklearn.model_selection import train_test_split
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv('NOB original data.csv')
X = data.drop('NOB', axis=1)
y = data['NOB']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

param_space_svm = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.2, 0.5, 1],
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [2, 3, 4],
}

def create_svm_model(C, epsilon, kernel, degree):
    return SVR(
        C=C,
        epsilon=epsilon,
        kernel=kernel,
        degree=degree
    )

svm_model = SVR()

opt_svm = BayesSearchCV(
    svm_model,
    param_space_svm,
    n_iter=80,
    cv=5,
    n_jobs=-1
)

opt_svm.fit(X_train, y_train)

best_params_svm = opt_svm.best_params_
print("Best SVR Hyperparameters:", best_params_svm)

best_svm_model = opt_svm.best_estimator_  
best_svm_model.fit(X_train, y_train)

y_pred1 = best_svm_model.predict(X_train).ravel()

y_pred = best_svm_model.predict(X_test)

y_test_flat1 = np.ravel(y_train)
y_pred_flat1 = np.ravel(y_pred1)
y_test_flat = np.ravel(y_test)
y_pred_flat = np.ravel(y_pred)

rmse1 = np.sqrt(mean_squared_error(y_train, y_pred1))
print("Train RMSE: {:.3f}".format(rmse1))
mae1 = mean_absolute_error(y_train, y_pred1)
print("Train MAE: {:.3f}".format(mae1))
r21 = r2_score(y_train, y_pred1)
print("Train R²: {:.3f}".format(r21))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Test RMSE: {:.3f}".format(rmse))
mae = mean_absolute_error(y_test, y_pred)
print("Test MAE: {:.3f}".format(mae))
r2 = r2_score(y_test, y_pred)
print("Test R²: {:.3f}".format(r2))

