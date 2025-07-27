#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Shap-XGBoost
shap.initjs()
def model_predict(data):
    return best_xgb_model.predict(data)

explainer = shap.TreeExplainer(model=best_xgb_model, data=X_train)
shap_values = explainer.shap_values(X_train, check_additivity=False)

feature_names = X.columns

mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
for feature, abs_shap_value in zip(feature_names, mean_abs_shap_values):
    print(f"{feature}\t\t{abs_shap_value:.3f}")

shap.summary_plot(shap_values, X_train, feature_names=X.columns, plot_type="bar")
shap.summary_plot(shap_values, X_train, feature_names=X.columns, cmap="PiYG")

shap_values_mean_expl = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=X_train)

X_train_original = scaler.inverse_transform(X_train)
X_test_original = scaler.inverse_transform(X_test)
explainer2 = shap.TreeExplainer(model=best_xgb_model, data=X_train_original)
shap_values2 = explainer2.shap_values(X_train_original, check_additivity=False)

shap_interaction_values = shap.TreeExplainer(best_xgb_model).shap_interaction_values(X_train_original)

shap.dependence_plot(("FA", "O2"), shap_interaction_values, X_train_original, show=False, feature_names=X.columns)

fa_idx = X.columns.get_loc("FA")
o2_idx = X.columns.get_loc("O2")

interaction_values_fa_o2 = shap_interaction_values[:, fa_idx, o2_idx]

interaction_df = pd.DataFrame({
    "Sample": range(len(interaction_values_fa_o2)),  
    "FA_O2_Interaction": interaction_values_fa_o2
})

print(interaction_df)

fa_values = X_train_original[:, fa_idx]  
o2_values = X_train_original[:, o2_idx]    

plt.figure(figsize=(7, 5))
scatter = plt.scatter(fa_values, interaction_values_fa_o2 * 2, c=o2_values, s=300, cmap='coolwarm', alpha=0.7, edgecolors='none')
cbar = plt.colorbar(scatter, pad=0.04, shrink=1, aspect=18)
plt.xlabel('FA (mg/L)', family='Times New Roman', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
cbar.ax.tick_params(labelsize=18)
plt.xlim(-8, 150)  
plt.axhline(0, color='black', linewidth=0.5, linestyle='-')
plt.show()

O2_column_index = 2 
FA_column_index = 3 
O2_values = X_test_original[:, O2_column_index]
Temperature_values = X_test_original[:, Temperature_column_index]
shap_data1 = shap_values1[:, O2_column_index]
shap_data2 = shap_values1[:, FA_column_index]
shap_data3 = shap_data5 + shap_data6

for O2_value, FA_value, shap_value in zip(O2_values, FA_values, shap_data3):
    print(f"O2: {O2_value:.4f}, FA: {FA_value:.4f}, SHAP value: {shap_value:.4f}")

data = {'O2': O2_values,
        'FA': FA_values,
        'SHAP Value': shap_data3}
shap.summary_plot(shap_values1, X_test, feature_names=X.columns, plot_type="bar")

shap.summary_plot(shap_values1, X_test, feature_names=X.columns, cmap="PiYG")

shap_values_mean_expl1 = shap.Explanation(values=shap_values1, base_values=explainer.expected_value, data=X_test)
shap.plots.bar(shap_values_mean_expl1, max_display=None)

min_value = np.min(shap_data3)
max_value = np.max(shap_data3)

mapped_min = 1
mapped_max = 500

mapped_values = (shap_data3 - min_value) / (max_value - min_value) * (mapped_max - mapped_min) + mapped_min

scale_factor = 1000  

sizes = mapped_values

fig = plt.figure()

scatter = plt.scatter(O2_values, FA_values, s=sizes, c=shap_data3, cmap='Spectral_r', alpha=0.5, edgecolors='none')

cbar = plt.colorbar(scatter, pad=0.04, shrink=1, aspect=15)

plt.xlabel('O2', family='Times New Roman', fontsize=15)
plt.ylabel('FA', family='Times New Roman', fontsize=15)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
cbar.ax.tick_params(labelsize=12)

plt.show()

#Shap-ANN
shap.initjs()
def model_predict(x):
    return best_model.predict(x)

explainer = shap.KernelExplainer(model=best_model, data=X_train)
shap_values = explainer.shap_values(X_train, check_additivity=False)
feature_names = X.columns
shap_values_mean = np.mean(shap_values, axis=0)
shap_values_mean1 = shap_values_mean.mean(axis=0)
mean_abs_shap_values = np.abs(shap_values_mean).mean(axis=0)
for feature, abs_shap_value in zip(feature_names, mean_abs_shap_values):
    print(f"{feature}\t\t{abs_shap_value:.3f}")
shap.summary_plot(shap_values_mean, X_train, feature_names=X.columns, plot_type="bar")
shap.summary_plot(shap_values_mean, X_train, feature_names=X.columns, cmap="PiYG")
shap_values_mean_expl = shap.Explanation(values=shap_values_mean, base_values=explainer.expected_value, data=X_train)
shap.plots.bar(shap_values_mean_expl, max_display=None)
X_train_original = scaler.inverse_transform(X_train)
X_test_original = scaler.inverse_transform(X_test)
explainer2 = shap.KernelExplainer(model=best_model, data=X_train_original)
shap_values2 = explainer2.shap_values(X_test_original, check_additivity=False)
shap_values_mean3 = np.mean(shap_values2, axis=0)

FA_column_index = 1 
pH_column_index = 5

shap_data1 = shap_values_mean3[:, FA_column_index]
shap_data2 = shap_values_mean3[:, pH_column_index]
shap_data3 = shap_data1 + shap_data2

for FA_value, pH_value, shap_value in zip(FA_values, pH_values, shap_data3):
    print(f"FA: {FA_value:.4f}, pH: {pH_value:.4f}, SHAP value: {shap_value:.4f}")

data = {'FA': FA_values,
        'pH': Temperature_values,
        'SHAP Value': shap_data3}

min_value = np.min(shap_data3)
max_value = np.max(shap_data3)
mapped_min = 1
mapped_max = 500
mapped_values = (shap_data3 - min_value) / (max_value - min_value) * (mapped_max - mapped_min) + mapped_min

scale_factor = 1000  
sizes = mapped_values
fig = plt.figure()
scatter = plt.scatter(FA_values, pH_values, s=sizes, c=shap_data3, cmap='Spectral_r', alpha=0.5, edgecolors='none')
cbar = plt.colorbar(scatter, pad=0.04, shrink=1, aspect=15)
plt.xlabel('FA (mg/L)', family='Times New Roman', fontsize=15)
plt.ylabel('pH', family='Times New Roman', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
cbar.ax.tick_params(labelsize=12)
plt.show()

