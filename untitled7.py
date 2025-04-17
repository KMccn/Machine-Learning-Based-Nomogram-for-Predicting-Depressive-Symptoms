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

state=1


# 设置字体和保存路径
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_excel(r'C:\Users\v\Desktop\jiamin\02.xlsx')

# 划分特征和目标变量
X = df.drop(['DepressiveSymptoms'], axis=1)
y = df['DepressiveSymptoms']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=state, stratify=y
)



# import smote_variants as sv
# from sklearn.datasets import make_classification
# from collections import Counter

 

# # 应用SMOTE-IPF
# oversampler = sv.SMOTE_IPF()
# X_train_resampled, y_train_resampled = oversampler.sample(X_train, y_train)

from sklearn.datasets import make_classification
from imblearn.over_sampling import KMeansSMOTE
from imblearn.over_sampling import ADASYN,SVMSMOTE,SMOTENC
from collections import Counter

# 设置随机种子以确保可复现
np.random.seed(state)

 

# 应用KMeans-SMOTE
 
# kmeans_smote = KMeansSMOTE(
#         random_state=state,
#         cluster_balance_threshold=0.01,  # 降低阈值
#         #n_clusters=10,                 # 增加簇数
#         #k_neighbors=3                 # 减少邻居数，适合稀疏数据
#     )

# X_train_resampled, y_train_resampled = kmeans_smote.fit_resample(X_train, y_train)
 
# 应用ADASYN
#adasyn = ADASYN(random_state=state)
#X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)

#smote_svm = SVMSMOTE(random_state=state, k_neighbors=3)
#X_train_resampled, y_train_resampled = smote_svm.fit_resample(X_train, y_train)

# 应用SMOTE-NC
#smote_nc = SMOTENC(categorical_features=[0,1], random_state=state)
#X_train_resampled, y_train_resampled = smote_nc.fit_resample(X_train, y_train)


# 使用 SMOTE 对训练集进行过采样
smote = SMOTE(random_state=state)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 查看数据分布
print("原始训练集类别分布:", dict(zip(*np.unique(y_train, return_counts=True))))
print("SMOTE 后训练集类别分布:", dict(zip(*np.unique(y_train_resampled, return_counts=True))))

# 初始化和训练模型
models = {
    "LightGBM": LGBMClassifier(random_state=state),
    "SVM": SVC(probability=True, random_state=state),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=state),
}

param_grids = {
    "LightGBM": {
        'n_estimators': [100, 200,300,500],           # 默认: 100（已包含）
        'learning_rate': [0.001,0.005,0.01],        # 默认: 0.1（已包含）
        'max_depth': [-1, 10, 20],                # 默认: -1（已包含）
        'num_leaves': [10,20,31, 50],              # 默认: 31（已包含）
        'min_child_samples': [20, 30,40,50],        # 默认: 20（已包含）
        #'subsample': [0.8, 1.0],                  # 新增，默认: 1.0
        #'colsample_bytree': [0.8, 1.0],           # 新增，默认: 1.0
        #'reg_alpha': [0.0, 0.1],                  # 新增，默认: 0.0
        #'reg_lambda': [0.0, 0.1],                 # 新增，默认: 0.0
    },
    "SVM": {
        'C': [0.1, 1.0, 10, 100],                 # 默认: 1.0（已包含）
        #'kernel': ['rbf', 'linear'],              # 新增，默认: 'rbf'
        #'gamma': ['scale', 'auto', 0.1],          # 默认: 'scale'（已包含）
    },
    "XGBoost": {
        'n_estimators': [50, 100, 200],           # 默认: 100（已包含）
        'learning_rate': [0.01, 0.1, 0.2],        # 默认: 0.1（已包含）
        'max_depth': [3, 5, 10],                  # 默认: 3（已包含）
        'subsample': [0.8, 1.0],                  # 默认: 1.0（已包含）
        'colsample_bytree': [0.8, 1.0],           # 默认: 1.0（已包含）
        'reg_alpha': [0.0, 0.1],                  # 新增，默认: 0.0
        'reg_lambda': [0.0, 1.0],                 # 新增，默认: 1.0
    },
}

# 使用 GridSearchCV 训练每个模型
# best_models = {}
# for name, model in models.items():
#     print(f"正在训练 {name} 模型...")
#     param_grid = param_grids[name]
#     grid_search = GridSearchCV(
#         estimator=model,
#         param_grid=param_grid,
#         scoring='neg_log_loss',
#         cv=5,
#         n_jobs=-1,
#         verbose=1,
#     )
#     grid_search.fit(X_train_resampled, y_train_resampled)
#     best_models[name] = grid_search.best_estimator_


# 定义多个评分指标
scoring = {
    'neg_log_loss': 'neg_log_loss',
    'neg_brier_score': make_scorer(lambda y, y_pred: -brier_score_loss(y, y_pred), needs_proba=True)
}

from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV

def train_model(name, model, param_grid, X_train, y_train):
    print(f"正在训练 {name} 模型...")
    # 使用多评分GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        refit='neg_brier_score',  # 选择用于最终模型的指标
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    return name, grid_search.best_estimator_

# 并行训练所有模型
best_models = {}
results = Parallel(n_jobs=-1, verbose=1)(
    delayed(train_model)(name, model, param_grids[name], X_train_resampled, y_train_resampled)
    for name, model in models.items()
)

# 将结果存入 best_models
for name, best_estimator in results:
    best_models[name] = best_estimator
    
    
    
    
# 绘制 ROC 曲线：分为训练集和测试集两个子图，并保存在同一画布上
fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # 创建一行两列的子图

# 遍历每个模型并计算性能指标
for name, model in best_models.items():
    # 在训练集上预测
    y_train_pred = model.predict(X_train)
    y_train_pred_prob = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") else None

    # 在测试集上预测
    y_test_pred = model.predict(X_test)
    y_test_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # 计算训练集 ROC 和 AUC
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_prob)
    auc_train = auc(fpr_train, tpr_train)

    # 计算测试集 ROC 和 AUC
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_prob)
    auc_test = auc(fpr_test, tpr_test)

    # 绘制训练集 ROC 曲线
    axes[0].plot(fpr_train, tpr_train, label=f"{name}: AUC={auc_train:.3f}")
    # 绘制测试集 ROC 曲线
    axes[1].plot(fpr_test, tpr_test, label=f"{name}: AUC={auc_test:.3f}")

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()

    # 计算指标
    sensitivity = tp / (tp + fn)  # 敏感性
    specificity = tn / (tn + fp)  # 特异性
    precision = tp / (tp + fp)  # 精确率
    accuracy = (tp + tn) / (tp + tn + fp + fn)  # 准确率
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)  # F1 得分
    kappa = cohen_kappa_score(y_test, y_test_pred)  # Kappa 值

    # 打印性能指标
    print(f"\n{name} 模型性能：")
    print(f"Sensitivity (Recall): {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Kappa: {kappa:.3f}")
    print(f"AUC: {auc_test:.3f}")

# 设置训练集子图的标题和标注
axes[0].set_title("Training Set ROC Curve", fontsize=16, fontweight="bold")
axes[0].set_xlabel("False Positive Rate (1-Specificity)", fontsize=14)
axes[0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=14)
axes[0].legend(loc="lower right", fontsize=12)
axes[0].grid(alpha=0.3)

# 设置测试集子图的标题和标注
axes[1].set_title("Testing Set ROC Curve", fontsize=16, fontweight="bold")
axes[1].set_xlabel("False Positive Rate (1-Specificity)", fontsize=14)
axes[1].set_ylabel("True Positive Rate (Sensitivity)", fontsize=14)
axes[1].legend(loc="lower right", fontsize=12)
axes[1].grid(alpha=0.3)

# 保存图形
output_path = r"C:\Users\v\Desktop\jiamin\ROC_Curve_Comparison_Train_Test.pdf"
plt.tight_layout()
plt.savefig(output_path, format='pdf', bbox_inches='tight')
plt.show()

print(f"\n训练集和测试集的 ROC 曲线已保存为 PDF 文件，路径为：{output_path}")

# 打印三个最佳模型的超参数
print("\n=== 最佳模型的超参数 ===")
for name, model in best_models.items():
    print(f"\n{name} 最佳模型超参数:")
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

# 绘制校准曲线：比较训练集和测试集
# 创建子图
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色和绿色

# 遍历每个模型并计算校准曲线
for i, (name, model) in enumerate(best_models.items()):
    # 获取模型预测概率
    y_train_pred_prob = model.predict_proba(X_train)[:, 1]
    y_test_pred_prob = model.predict_proba(X_test)[:, 1]

    # 计算训练集校准曲线
    prob_true_train, prob_pred_train = calibration_curve(y_train, y_train_pred_prob, n_bins=10)

    # 计算测试集校准曲线
    prob_true_test, prob_pred_test = calibration_curve(y_test, y_test_pred_prob, n_bins=10)

    # 绘制训练集校准曲线
    axes[0].plot(prob_pred_train, prob_true_train, marker='o', linewidth=2,
                 label=f"{name}", color=colors[i])

    # 绘制测试集校准曲线
    axes[1].plot(prob_pred_test, prob_true_test, marker='o', linewidth=2,
                 label=f"{name}", color=colors[i])

# 在两个子图上都绘制对角线（表示完美校准）
for ax in axes:
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    ax.set_xlabel('Mean Predicted Probability', fontsize=14)
    ax.set_ylabel('Fraction of Positives', fontsize=14)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.legend(loc='best', fontsize=12)
    ax.grid(alpha=0.3)

# 设置图表标题
axes[0].set_title('Calibration Curve (Training Set)', fontsize=16, fontweight='bold')
axes[1].set_title('Calibration Curve (Testing Set)', fontsize=16, fontweight='bold')

# 保存图形
output_path_calibration = r"C:\Users\v\Desktop\jiamin\Calibration_Curves_Train_Test.pdf"
plt.tight_layout()
plt.savefig(output_path_calibration, format='pdf', bbox_inches='tight')
plt.show()

print(f"\n训练集和测试集的校准曲线已保存为 PDF 文件，路径为：{output_path_calibration}")


# 获取最佳LightGBM模型
lightgbm_model = best_models["LightGBM"]

# 初始化SHAP解释器
explainer = shap.TreeExplainer(lightgbm_model)

# 计算SHAP值
shap_values_numpy = explainer.shap_values(X)

# 如果返回的是一个列表（多分类情况下），取阳性类的SHAP值
if isinstance(shap_values_numpy, list):
    shap_values_numpy = shap_values_numpy[1]  # 假设第二个元素是阳性类的SHAP值

# 转换为DataFrame
shap_values_df = pd.DataFrame(shap_values_numpy, columns=X.columns)
print("SHAP值的前几行样例:")
print(shap_values_df.head())

# 计算SHAP值的绝对值
shap_values_abs = shap_values_df.abs()

# 根据原始数据df['DepressiveSymptoms']分组，计算特征贡献度的绝对值均值
mean_abs_contributions = shap_values_abs.groupby(df['DepressiveSymptoms']).mean()
mean_abs_contributions_transposed = mean_abs_contributions.T

# 为mean_abs_contributions_transposed添加一列，用于存储每个特征的mean(|SHAP value|)均值
mean_abs_contributions_transposed['mean_contribution'] = mean_abs_contributions_transposed.mean(axis=1)

# 根据'mean_contribution'列进行升序排序（从底部到顶部显示）
sorted_contributions = mean_abs_contributions_transposed.sort_values(by='mean_contribution', ascending=True)

# 删除排序辅助列，保留排序后的结果
sorted_contributions = sorted_contributions.drop(columns=['mean_contribution'])

# 打印排序后的结果
print("\n按贡献度排序的特征:")
print(sorted_contributions)

# 准备数据
features = sorted_contributions.index  # 特征名
class_0_values = sorted_contributions[0]  # 类别0的均值（阴性类）
class_1_values = sorted_contributions[1]  # 类别1的均值（阳性类）

# 计算误差线的上下限
error_0 = shap_values_abs[df['DepressiveSymptoms'] == 0].std()  # 类别0的标准差作为误差
error_1 = shap_values_abs[df['DepressiveSymptoms'] == 1].std()  # 类别1的标准差作为误差

# 开始绘制
fig, ax = plt.subplots(figsize=(10, 14))

# 绘制类别0（左侧，蓝色），通过负方向偏移实现显示在左侧，同时数值为正
ax.barh(features, -class_0_values, color="skyblue", edgecolor="black", label="无抑郁")
ax.errorbar(-class_0_values, features, xerr=[error_0, np.zeros_like(error_0)],  # 左侧误差线
            fmt="none", ecolor="black", capsize=4, elinewidth=1.2)

# 绘制类别1（右侧，紫色），保持右侧正值
ax.barh(features, class_1_values, color="purple", edgecolor="black", label="有抑郁")
ax.errorbar(class_1_values, features, xerr=[np.zeros_like(error_1), error_1],  # 右侧误差线
            fmt="none", ecolor="black", capsize=4, elinewidth=1.2)

# 添加竖线：x=0用实线
ax.axvline(0, color="black", linewidth=1.2, linestyle="-")  # x=0中心实线

# 自定义x轴刻度，将类别0的区域映射为正值
max_value = max(class_0_values.max(), class_1_values.max()) * 1.1  # 获取最大值并稍微扩大范围
x_ticks = np.linspace(0, max_value, 5)  # 自定义刻度范围（正值）
x_ticks_negative = -x_ticks[1:]  # 创建左侧对称刻度（负值跳过0）
ax.set_xticks(np.concatenate([x_ticks_negative, x_ticks]))  # 合并正负刻度
ax.set_xticklabels([f"{abs(x):.2f}" for x in np.concatenate([x_ticks_negative, x_ticks])])  # 设置标签为正值，保留2位小数

# 灰色虚线优化，左侧和右侧对称
for x in x_ticks:
    if x != 0:
        ax.axvline(-x, color="#696969", linewidth=0.8, linestyle="--", zorder=0)  # 左侧灰色虚线
        ax.axvline(x, color="#696969", linewidth=0.8, linestyle="--", zorder=0)  # 右侧灰色虚线

# 字体整体加粗
ax.set_xlabel("平均 |SHAP| 值", fontsize=14, fontweight='bold')  # x轴标签加粗
ax.set_ylabel("特征", fontsize=14, fontweight='bold')  # y轴标签加粗
ax.set_title("按类别划分的特征贡献度 (LightGBM模型)", fontsize=16, fontweight='bold')  # 标题加粗

# 设置图例字体加粗
ax.legend(loc="lower right", fontsize=12, frameon=False, prop={'weight': 'bold'})  # 图例右下角，无边框，加粗

# 设置x和y轴刻度字体加粗
ax.tick_params(axis='both', which='major', labelsize=12, width=1.2, length=6)  # 刻度线加粗

# 设置x轴范围
ax.set_xlim(-max_value, max_value)  # 对称的x轴范围

# 保证布局紧凑
plt.tight_layout()

# 保存为PDF格式
output_path_pdf = r"C:\Users\v\Desktop\jiamin\LightGBM_SHAP_Feature_Contributions.pdf"
plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
plt.show()

print(f"\nSHAP特征贡献度图已保存为PDF: {output_path_pdf}")




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_excel(r'C:\Users\v\Desktop\jiamin\03.xlsx')

# 划分特征和目标变量
X = df.drop(['DepressiveSymptoms'], axis=1)
y = df['DepressiveSymptoms']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=state, stratify=df['DepressiveSymptoms']
)

# 使用 SMOTE 对训练集进行过采样
smote = SMOTE(random_state=state)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 查看数据分布
print("原始训练集类别分布:", dict(zip(*np.unique(y_train, return_counts=True))))
print("SMOTE 后训练集类别分布:", dict(zip(*np.unique(y_train_resampled, return_counts=True))))

# 使用第一段代码中训练好的LightGBM模型
lightgbm_model = best_models["LightGBM"]

# 获取模型的最佳参数
best_params = lightgbm_model.get_params()
print("\n使用的最佳参数:")
for param in ['n_estimators', 'learning_rate', 'max_depth', 'num_leaves', 'min_child_samples']:
    print(f"  {param}: {best_params[param]}")

# 定义要测试的特征组合
feature_combinations = [
    ['Age', 'Exercise', 'ChronicDisease', 'Insomnia', 'AnxietySymptoms'],  # 所有特征
    ['Exercise', 'ChronicDisease', 'Insomnia', 'AnxietySymptoms'],         # 无Age
    ['Age', 'ChronicDisease', 'Insomnia', 'AnxietySymptoms'],             # 无Exercise
    ['Age', 'Exercise', 'Insomnia', 'AnxietySymptoms'],                   # 无ChronicDisease
    ['Age', 'Exercise', 'ChronicDisease', 'AnxietySymptoms'],             # 无Insomnia
    ['Age', 'Exercise', 'ChronicDisease', 'Insomnia']                     # 无AnxietySymptoms
]

# 为图例准备的标签
feature_labels = [
    'All Features',
    'No Age',
    'No Exercise',
    'No ChronicDisease',
    'No Insomnia',
    'No AnxietySymptoms'
]

# 创建图形
plt.figure(figsize=(10, 8))

# 为每个特征组合训练模型并绘制ROC曲线
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
aucs = []

for i, features in enumerate(feature_combinations):
    # 使用所选特征提取数据
    X_train_subset = X_train_resampled[features]  # 这里使用了SMOTE过采样后的训练集
    X_test_subset = X_test[features]
    
    # 使用最佳参数创建新的LightGBM模型
    # 注意：我们需要为每个特征子集创建一个新模型，因为特征数量不同
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        num_leaves=best_params['num_leaves'],
        min_child_samples=best_params['min_child_samples'],
        random_state=state
    )
    
    # 训练模型
    model.fit(X_train_subset, y_train_resampled)  # 注意：这里使用了SMOTE过采样后的标签
    
    # 在测试集上进行预测
    y_pred_proba = model.predict_proba(X_test_subset)[:, 1]
    
    # 计算ROC曲线点
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    # 计算AUC
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    
    # 绘制ROC曲线
    plt.plot(fpr, tpr, color=colors[i], lw=2, 
             label=f'{feature_labels[i]} (AUC = {roc_auc:.3f})')

# 绘制随机分类器的ROC曲线
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.7)

# 设置图表属性
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1-Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ROC Curves for Different Feature Combinations', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)

# 保存图表
plt.tight_layout()
plt.savefig(r'C:\Users\v\Desktop\jiamin\LightGBM_ROC_Feature_Combinations.pdf', format='pdf', bbox_inches='tight')
plt.savefig(r'C:\Users\v\Desktop\jiamin\LightGBM_ROC_Feature_Combinations.png', format='png', dpi=300, bbox_inches='tight')

plt.show()
