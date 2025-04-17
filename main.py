# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 11:42:30 2025

@author: v
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 19:28:51 2025

@author: v
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, cohen_kappa_score
from sklearn.calibration import calibration_curve  # Correct import for calibration_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import warnings
from sklearn.metrics import make_scorer, log_loss, brier_score_loss
import shap

warnings.filterwarnings("ignore")

state = 1

# Set font and save path
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# Read data
df = pd.read_excel(r'C:\Users\v\Desktop\jiamin\02.xlsx')

# Split features and target variable
X = df.drop(['DepressiveSymptoms'], axis=1)
y = df['DepressiveSymptoms']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=state, stratify=y
)

from sklearn.datasets import make_classification
from imblearn.over_sampling import KMeansSMOTE
from imblearn.over_sampling import ADASYN, SVMSMOTE, SMOTENC
from collections import Counter

# Set random seed for reproducibility
np.random.seed(state)

# Apply SMOTE to oversample the training set
smote = SMOTE(random_state=state)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check data distribution
print("Original training set class distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
print("SMOTE training set class distribution:", dict(zip(*np.unique(y_train_resampled, return_counts=True))))

# Initialize and train models
models = {
    "LightGBM": LGBMClassifier(random_state=state),
    "SVM": SVC(probability=True, random_state=state),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=state),
}

param_grids = {
    "LightGBM": {
        'n_estimators': [100, 200, 300, 500],           # Default: 100 (included)
        'learning_rate': [0.001, 0.005, 0.01],          # Default: 0.1 (included)
        'max_depth': [-1, 10, 20],                      # Default: -1 (included)
        'num_leaves': [10, 20, 31, 50],                 # Default: 31 (included)
        'min_child_samples': [20, 30, 40, 50],          # Default: 20 (included)
    },
    "SVM": {
        'C': [0.1, 1.0, 10, 100],                       # Default: 1.0 (included)
    },
    "XGBoost": {
        'n_estimators': [50, 100, 200],                 # Default: 100 (included)
        'learning_rate': [0.01, 0.1, 0.2],              # Default: 0.1 (included)
        'max_depth': [3, 5, 10],                        # Default: 3 (included)
        'subsample': [0.8, 1.0],                        # Default: 1.0 (included)
        'colsample_bytree': [0.8, 1.0],                 # Default: 1.0 (included)
        'reg_alpha': [0.0, 0.1],                        # Added, Default: 0.0
        'reg_lambda': [0.0, 1.0],                       # Added, Default: 1.0
    },
}

# Define multiple scoring metrics
scoring = {
    'neg_log_loss': 'neg_log_loss',
    'neg_brier_score': make_scorer(lambda y, y_pred: -brier_score_loss(y, y_pred), needs_proba=True)
}

from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV

def train_model(name, model, param_grid, X_train, y_train):
    print(f"Training {name} model...")
    # Use GridSearchCV with multiple scoring metrics
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        refit='neg_brier_score',  # Metric used for final model selection
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    return name, grid_search.best_estimator_

# Train all models in parallel
best_models = {}
results = Parallel(n_jobs=-1, verbose=1)(
    delayed(train_model)(name, model, param_grids[name], X_train_resampled, y_train_resampled)
    for name, model in models.items()
)

# Store results in best_models
for name, best_estimator in results:
    best_models[name] = best_estimator

# Plot ROC curves: Separate training and testing sets in two subplots on the same canvas
fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # Create a 1x2 subplot grid

# Iterate through each model and calculate performance metrics
for name, model in best_models.items():
    # Predict on training set
    y_train_pred = model.predict(X_train)
    y_train_pred_prob = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") else None

    # Predict on testing set
    y_test_pred = model.predict(X_test)
    y_test_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Calculate ROC and AUC for training set
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_prob)
    auc_train = auc(fpr_train, tpr_train)

    # Calculate ROC and AUC for testing set
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_prob)
    auc_test = auc(fpr_test, tpr_test)

    # Plot ROC curve for training set
    axes[0].plot(fpr_train, tpr_train, label=f"{name}: AUC={auc_train:.3f}")
    # Plot ROC curve for testing set
    axes[1].plot(fpr_test, tpr_test, label=f"{name}: AUC={auc_test:.3f}")

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics
    sensitivity = tp / (tp + fn)  # Sensitivity
    specificity = tn / (tn + fp)  # Specificity
    precision = tp / (tp + fp)  # Precision
    accuracy = (tp + tn) / (tp + tn + fp + fn)  # Accuracy
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)  # F1 Score
    kappa = cohen_kappa_score(y_test, y_test_pred)  # Kappa value

    # Print performance metrics
    print(f"\n{name} model performance:")
    print(f"Sensitivity (Recall): {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Kappa: {kappa:.3f}")
    print(f"AUC: {auc_test:.3f}")

# Set title and labels for training set subplot
axes[0].set_title("Training Set ROC Curve", fontsize=16, fontweight="bold")
axes[0].set_xlabel("False Positive Rate (1-Specificity)", fontsize=14)
axes[0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=14)
axes[0].legend(loc="lower right", fontsize=12)
axes[0].grid(alpha=0.3)

# Set title and labels for testing set subplot
axes[1].set_title("Testing Set ROC Curve", fontsize=16, fontweight="bold")
axes[1].set_xlabel("False Positive Rate (1-Specificity)", fontsize=14)
axes[1].set_ylabel("True Positive Rate (Sensitivity)", fontsize=14)
axes[1].legend(loc="lower right", fontsize=12)
axes[1].grid(alpha=0.3)

# Save the figure
output_path = r"C:\Users\v\Desktop\jiamin\ROC_Curve_Comparison_Train_Test.pdf"
plt.tight_layout()
plt.savefig(output_path, format='pdf', bbox_inches='tight')
plt.show()

print(f"\nROC curves for training and testing sets have been saved as a PDF file at: {output_path}")

# Print hyperparameters of the best models
print("\n=== Best Model Hyperparameters ===")
for name, model in best_models.items():
    print(f"\n{name} Best Model Hyperparameters:")
    if name == "LightGBM":
        print(f"  n_estimators: {model.n_estimators}")
        print(f"  learning_rate: {model.learning_rate}")
        print(f"  max_depth: {model.max_depth}")
        print(f"  num_leaves: {model.num_leaves}")
        print(f"  min_child_samples: {model.min_child_samples}")
    elif name == "SVM":
        print(f"  C: {model.C}")
        print(f"  kernel: {model.kernel}")
        print(f"  gamma: {model.gamma}")
    elif name == "XGBoost":
        print(f"  n_estimators: {model.n_estimators}")
        print(f"  learning_rate: {model.learning_rate}")
        print(f"  max_depth: {model.max_depth}")
        print(f"  subsample: {model.subsample}")
        print(f"  colsample_bytree: {model.colsample_bytree}")

# Plot calibration curves: Compare training and testing sets
# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, and green

# Iterate through each model and calculate calibration curves
for i, (name, model) in enumerate(best_models.items()):
    # Get predicted probabilities
    y_train_pred_prob = model.predict_proba(X_train)[:, 1]
    y_test_pred_prob = model.predict_proba(X_test)[:, 1]

    # Calculate calibration curve for training set
    prob_true_train, prob_pred_train = calibration_curve(y_train, y_train_pred_prob, n_bins=10)

    # Calculate calibration curve for testing set
    prob_true_test, prob_pred_test = calibration_curve(y_test, y_test_pred_prob, n_bins=10)

    # Plot calibration curve for training set
    axes[0].plot(prob_pred_train, prob_true_train, marker='o', linewidth=2,
                 label=f"{name}", color=colors[i])

    # Plot calibration curve for testing set
    axes[1].plot(prob_pred_test, prob_true_test, marker='o', linewidth=2,
                 label=f"{name}", color=colors[i])

# Draw diagonal line (representing perfect calibration) on both subplots
for ax in axes:
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    ax.set_xlabel('Mean Predicted Probability', fontsize=14)
    ax.set_ylabel('Fraction of Positives', fontsize=14)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.legend(loc='best', fontsize=12)
    ax.grid(alpha=0.3)

# Set chart titles
axes[0].set_title('Calibration Curve (Training Set)', fontsize=16, fontweight='bold')
axes[1].set_title('Calibration Curve (Testing Set)', fontsize=16, fontweight='bold')

# Save the figure
output_path_calibration = r"C:\Users\v\Desktop\jiamin\Calibration_Curves_Train_Test.pdf"
plt.tight_layout()
plt.savefig(output_path_calibration, format='pdf', bbox_inches='tight')
plt.show()

print(f"\nCalibration curves for training and testing sets have been saved as a PDF file at: {output_path_calibration}")

# Get the best LightGBM model
lightgbm_model = best_models["LightGBM"]

# Initialize SHAP explainer
explainer = shap.TreeExplainer(lightgbm_model)

# Calculate SHAP values
shap_values_numpy = explainer.shap_values(X)

# If a list is returned (for multi-class cases), take SHAP values for the positive class
if isinstance(shap_values_numpy, list):
    shap_values_numpy = shap_values_numpy[1]  # Assume the second element is for the positive class

# Convert to DataFrame
shap_values_df = pd.DataFrame(shap_values_numpy, columns=X.columns)
print("Sample of SHAP values (first few rows):")
print(shap_values_df.head())

# Calculate absolute SHAP values
shap_values_abs = shap_values_df.abs()

# Group by df['DepressiveSymptoms'] and calculate the mean of absolute feature contributions
mean_abs_contributions = shap_values_abs.groupby(df['DepressiveSymptoms']).mean()
mean_abs_contributions_transposed = mean_abs_contributions.T

# Add a column for the mean of mean(|SHAP value|) across classes
mean_abs_contributions_transposed['mean_contribution'] = mean_abs_contributions_transposed.mean(axis=1)

# Sort by 'mean_contribution' column in ascending order (display from bottom to top)
sorted_contributions = mean_abs_contributions_transposed.sort_values(by='mean_contribution', ascending=True)

# Drop the sorting helper column, keeping the sorted result
sorted_contributions = sorted_contributions.drop(columns=['mean_contribution'])

# Print sorted results
print("\nFeatures sorted by contribution:")
print(sorted_contributions)

# Prepare data
features = sorted_contributions.index  # Feature names
class_0_values = sorted_contributions[0]  # Mean values for class 0 (negative class)
class_1_values = sorted_contributions[1]  # Mean values for class 1 (positive class)

# Calculate error bar limits
error_0 = shap_values_abs[df['DepressiveSymptoms'] == 0].std()  # Standard deviation for class 0 as error
error_1 = shap_values_abs[df['DepressiveSymptoms'] == 1].std()  # Standard deviation for class 1 as error

# Start plotting
fig, ax = plt.subplots(figsize=(10, 14))

# Plot class 0 (left side, blue), using negative offset to display on the left with positive values
ax.barh(features, -class_0_values, color="skyblue", edgecolor="black", label="No Depression")
ax.errorbar(-class_0_values, features, xerr=[error_0, np.zeros_like(error_0)],  # Left-side error bars
            fmt="none", ecolor="black", capsize=4, elinewidth=1.2)

# Plot class 1 (right side, purple), keeping positive values on the right
ax.barh(features, class_1_values, color="purple", edgecolor="black", label="Depression")
ax.errorbar(class_1_values, features, xerr=[np.zeros_like(error_1), error_1],  # Right-side error bars
            fmt="none", ecolor="black", capsize=4, elinewidth=1.2)

# Add vertical line: x=0 with solid line
ax.axvline(0, color="black", linewidth=1.2, linestyle="-")  # Solid center line at x=0

# Customize x-axis ticks, mapping class 0 region to positive values
max_value = max(class_0_values.max(), class_1_values.max()) * 1.1  # Get max value and slightly expand range
x_ticks = np.linspace(0, max_value, 5)  # Define tick range (positive values)
x_ticks_negative = -x_ticks[1:]  # Create symmetric left-side ticks (skip 0)
ax.set_xticks(np.concatenate([x_ticks_negative, x_ticks]))  # Combine positive and negative ticks
ax.set_xticklabels([f"{abs(x):.2f}" for x in np.concatenate([x_ticks_negative, x_ticks])])  # Set labels as positive values, 2 decimal places

# Add gray dashed lines for optimization, symmetric on left and right
for x in x_ticks:
    if x != 0:
        ax.axvline(-x, color="#696969", linewidth=0.8, linestyle="--", zorder=0)  # Left gray dashed line
        ax.axvline(x, color="#696969", linewidth=0.8, linestyle="--", zorder=0)  # Right gray dashed line

# Bold fonts throughout
ax.set_xlabel("Mean |SHAP| Value", fontsize=14, fontweight='bold')  # Bold x-axis label
ax.set_ylabel("Feature", fontsize=14, fontweight='bold')  # Bold y-axis label
ax.set_title("Feature Contributions by Class (LightGBM Model)", fontsize=16, fontweight='bold')  # Bold title

# Set legend font to bold
ax.legend(loc="lower right", fontsize=12, frameon=False, prop={'weight': 'bold'})  # Legend in bottom right, no frame, bold

# Set x and y axis tick fonts to bold
ax.tick_params(axis='both', which='major', labelsize=12, width=1.2, length=6)  # Bold tick lines

# Set x-axis range
ax.set_xlim(-max_value, max_value)  # Symmetric x-axis range

# Ensure tight layout
plt.tight_layout()

# Save as PDF
output_path_pdf = r"C:\Users\v\Desktop\jiamin\LightGBM_SHAP_Feature_Contributions.pdf"
plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
plt.show()

print(f"\nSHAP feature contribution plot has been saved as PDF: {output_path_pdf}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Set font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# Read data
df = pd.read_excel(r'C:\Users\v\Desktop\jiamin\03.xlsx')

# Split features and target variable
X = df.drop(['DepressiveSymptoms'], axis=1)
y = df['DepressiveSymptoms']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=state, stratify=df['DepressiveSymptoms']
)

# Apply SMOTE to oversample the training set
smote = SMOTE(random_state=state)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check data distribution
print("Original training set class distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
print("SMOTE training set class distribution:", dict(zip(*np.unique(y_train_resampled, return_counts=True))))

# Use the trained LightGBM model from the first section
lightgbm_model = best_models["LightGBM"]

# Get the best parameters of the model
best_params = lightgbm_model.get_params()
print("\nBest parameters used:")
for param in ['n_estimators', 'learning_rate', 'max_depth', 'num_leaves', 'min_child_samples']:
    print(f"  {param}: {best_params[param]}")

# Define feature combinations to test
feature_combinations = [
    ['Age', 'Exercise', 'ChronicDisease', 'Insomnia', 'AnxietySymptoms'],  # All features
    ['Exercise', 'ChronicDisease', 'Insomnia', 'AnxietySymptoms'],         # No Age
    ['Age', 'ChronicDisease', 'Insomnia', 'AnxietySymptoms'],             # No Exercise
    ['Age', 'Exercise', 'Insomnia', 'AnxietySymptoms'],                   # No ChronicDisease
    ['Age', 'Exercise', 'ChronicDisease', 'AnxietySymptoms'],             # No Insomnia
    ['Age', 'Exercise', 'ChronicDisease', 'Insomnia']                     # No AnxietySymptoms
]

# Prepare labels for the legend
feature_labels = [
    'All Features',
    'No Age',
    'No Exercise',
    'No ChronicDisease',
    'No Insomnia',
    'No AnxietySymptoms'
]

# Create figure
plt.figure(figsize=(10, 8))

# Train model and plot ROC curve for each feature combination
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
aucs = []

for i, features in enumerate(feature_combinations):
    # Extract data for selected features
    X_train_subset = X_train_resampled[features]  # Use SMOTE oversampled training set
    X_test_subset = X_test[features]
    
    # Create new LightGBM model with best parameters
    # Note: A new model is needed for each feature subset due to different feature counts
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        num_leaves=best_params['num_leaves'],
        min_child_samples=best_params['min_child_samples'],
        random_state=state
    )
    
    # Train the model
    model.fit(X_train_subset, y_train_resampled)  # Note: Use SMOTE oversampled labels
    
    # Predict on testing set
    y_pred_proba = model.predict_proba(X_test_subset)[:, 1]
    
    # Calculate ROC curve points
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    # Calculate AUC
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color=colors[i], lw=2, 
             label=f'{feature_labels[i]} (AUC = {roc_auc:.3f})')

# Plot ROC curve for random classifier
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.7)

# Set chart properties
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1-Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ROC Curves for Different Feature Combinations', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)

# Save the chart
plt.tight_layout()
plt.savefig(r'C:\Users\v\Desktop\jiamin\LightGBM_ROC_Feature_Combinations.pdf', format='pdf', bbox_inches='tight')
plt.savefig(r'C:\Users\v\Desktop\jiamin\LightGBM_ROC_Feature_Combinations.png', format='png', dpi=300, bbox_inches='tight')

plt.show()
