# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score



# %%
# Read the 'merged_df_train_4.pkl' pickle file and load it into the 'train' DataFrame
merged_df_train_4 = pd.read_pickle('merged_df_train_MAE.pkl')
df_train = pd.DataFrame(merged_df_train_4)

# Read the 'test_3.pkl' pickle file and load it into the 'test' DataFrame
merged_df_test_4 = pd.read_pickle('merged_df_test_MAE.pkl')
df_test = pd.DataFrame(merged_df_test_4)

# %%
merged_df_train_4

# %%
merged_df_test_4

# %%
# Compare the predicted labels from a model with the true labels
# Parameter true values and predicted values
confusion = confusion_matrix(df_train['isFraud'], df_train['isFraud_xgb'])
print(confusion)
# Confusion Matrix
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix after Threshold Adjustment')
plt.show()


# %% [markdown]
# TRAIN

# %%
conf_matrix = np.array(confusion)

# Thresholds
thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Initialize lists to store TPR and FPR for each threshold
tpr_list = []
fpr_list = []

for threshold in thresholds:
    # You would replace this part with your actual model predictions
    predicted_positive = (df_train['isFraud_xgb_proba'] >= threshold).astype(int)

    # Confusion matrix for the current threshold
    conf_matrix_threshold = confusion_matrix(df_train['isFraud'], predicted_positive)

    # Calculate True Positive Rate and False Positive Rate for the current threshold
    TPR = conf_matrix_threshold[1, 1] / (conf_matrix_threshold[1, 1] + conf_matrix_threshold[1, 0])
    FPR = conf_matrix_threshold[0, 1] / (conf_matrix_threshold[0, 0] + conf_matrix_threshold[0, 1])

    # Append TPR and FPR to the lists
    tpr_list.append(TPR)
    fpr_list.append(FPR)

# Print the results
for i, threshold in enumerate(thresholds):
    print(f"Threshold: {threshold}, TPR: {tpr_list[i]}, FPR: {fpr_list[i]}")

# Plot TPR vs. FPR
tpr_list = np.array(tpr_list)
fpr_list = np.array(fpr_list)

optimal_threshold_index = np.argmax(tpr_list - fpr_list)
optimal_threshold = thresholds[optimal_threshold_index]

# Plot TPR vs. FPR
plt.figure(figsize=(8, 8))
plt.plot(fpr_list, tpr_list, label='ROC Curve')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing')

# Labeling and customization
plt.title('ROC Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend()
plt.grid(True)
plt.show()




# %% [markdown]
# Find the index of the threshold where the ROC curve is closest to the upper left corner (0, 1), 
# which is considered an optimal point for a binary classification model. 

# %%
# 'isFraud' is the column with actual labels
# 'isFraud_xgb_proba' is the column with predicted probabilities

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(df_train['isFraud'], df_train['isFraud_xgb_proba'])

# Calculate AUC-ROC
roc_auc = roc_auc_score(df_train['isFraud'], df_train['isFraud_xgb_proba'])

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Find the optimal threshold
# np.argmax() -  returns the index of the maximum value in an array
# find the index of the threshold where the ROC curve is closest to the upper left corner (0, 1), which is considered an optimal point for a binary classification model. 
optimal_threshold_index = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_threshold_index]
print("optimal_threshold_index: ", optimal_threshold_index)
# This threshold is chosen as the optimal threshold for making predictions based on the ROC curve analysis
print('Optimal Threshold:', optimal_threshold)

df_train['isFraud_xgb_adjusted'] = (df_train['isFraud_xgb_proba'] >= optimal_threshold).astype(int)

# Calculate AUC directly using roc_auc_score
auc_score = roc_auc_score(df_train['isFraud'], df_train['isFraud_xgb_adjusted'])
print(f'AUC Score: {auc_score:.4f}')
print()
#classification report
print(classification_report(df_train['isFraud'], df_train['isFraud_xgb_adjusted']))

#confusion matrix
print(confusion_matrix(df_train['isFraud'], df_train['isFraud_xgb_adjusted'], normalize='true'))

confusion = confusion_matrix(df_train['isFraud'], df_train['isFraud_xgb_adjusted'])
print(confusion)
# Confusion Matrix
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix after Threshold Adjustment')
plt.show()


# %% [markdown]
# Calculate True Positive Rate and False Positive Rate for each threshold

# %% [markdown]
# TEST

# %%
# 'isFraud_xgb' is the column with original binary predictions
# 'isFraud_xgb_proba' is the column with predicted probabilities

# Adjust the classification threshold
adjusted_threshold = 0.17607044
df_test['isFraud_xgb_adjusted'] = (df_test['isFraud_xgb_proba'] >= adjusted_threshold).astype(int)


print("Threshold: ", adjusted_threshold)
# Calculate AUC directly using roc_auc_score
auc_score = roc_auc_score(df_test['isFraud'], df_test['isFraud_xgb_adjusted'])
print(f'AUC Score: {auc_score:.4f}')
print()
#classification report
print(classification_report(df_test['isFraud'], df_test['isFraud_xgb_adjusted']))

#confusion matrix
print(confusion_matrix(df_test['isFraud'], df_test['isFraud_xgb_adjusted'], normalize='true'))






