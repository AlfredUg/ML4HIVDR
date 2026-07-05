# --- Memory safety: cap BLAS thread count BEFORE numpy is imported ---
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from xgboost import XGBClassifier
import joblib

# =====================================================================
# 1. DATA SETUP 
# =====================================================================
np.random.seed(42)

variants_data = pd.read_csv("losAlomosData4ML.csv")
variants_data_full = variants_data

stanford_muts = pd.read_csv('data/Stanford-resistance-comments.csv')
stanford_mut_list = set(stanford_muts['Mutation'])

# Select only columns that are not in the exclude_cols list
selected_cols = variants_data.columns.difference(stanford_mut_list)

# Create a new DataFrame with only the selected columns
variants_data = variants_data[selected_cols]
# print(variants_data.shape)

variants_data = variants_data.loc[:, ~variants_data.columns.str.contains(r'[\*\-]')]

target = 'Status'
pos_label = 'yes'  
gene = 'Integrase'  

pure_mutation_pattern = r'^[A-Z][0-9]+[A-Z]$'
mixed_variants_data = variants_data.loc[:, ~variants_data.columns.str.match(pure_mutation_pattern)]
single_mut_variants_data = variants_data.loc[:, variants_data.columns.str.match(pure_mutation_pattern)]

X = single_mut_variants_data 
X_matrix = X.values  
n_samples, n_features = X_matrix.shape

print(f"Genomic Feature Matrix Shape: {X.shape}")

# Convert target string labels ("yes"/"no") to numeric (1/0) for XGBoost/Metrics stability
y_raw = variants_data[target].values.ravel()
y = (y_raw == pos_label).astype(int)

variants_data_full['subtype2'] = np.where(variants_data_full['Subtype'] == 'B', 'B', 'non-B')
subtypes = variants_data_full['subtype2'].values.ravel()
stratify_labels = np.array([f"{t}_{s}" for t, s in zip(y, subtypes)])
feature_names = X.columns.values

# =====================================================================
# 2. MODEL ENGINE CONFIGURATION
# =====================================================================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Linear SVM': LinearSVC(max_iter=10000, dual=False, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'GBM': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'Naive Bayes': GaussianNB()
}

param_grids = {
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'Linear SVM': {'C': [0.01, 0.1, 1, 10]},
    'Decision Tree': {'max_depth': [3, 5, 10]},
    'Random Forest': {'n_estimators': [50, 100], 'max_depth': [5, 10]},
    'GBM': {'learning_rate': [0.01, 0.1], 'n_estimators': [50, 100]},
    'XGBoost': {'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
    'Naive Bayes': {}
}

roc_plotting_data = {name: {'fpr_list': [], 'tpr_list': [], 'aucs': []} for name in models}
model_confusion_matrices = {name: np.zeros((2, 2)) for name in models}
model_performance_summary = {}

performance_records = []
importance_records = []

# =====================================================================
# 3. EXECUTE NESTED PIPELINE LOOP
# =====================================================================
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Starting Nested Cross-Validation Pipeline...\n")

for model_name, model in models.items():
    print("="*80)
    print(f"PROCESSING RUNS FOR ALGORITHM: {model_name}")
    print("="*80)
    
    fold_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(cv_outer.split(X_matrix, stratify_labels), 1):
        X_train_raw, X_val_raw = X_matrix[train_idx], X_matrix[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)
        
        # --- FEATURE SELECTION & COUNTS ---
        features_before = X_train_raw.shape[1]
        
        lasso_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        lasso_selector = LassoCV(cv=lasso_cv, random_state=42, max_iter=5000)
        lasso_selector.fit(X_train_scaled, y_train)
        
        selected_indices = np.where(lasso_selector.coef_ != 0)[0]
        if len(selected_indices) == 0: 
            selected_indices = np.arange(n_features)
            
        X_train_selected = X_train_scaled[:, selected_indices]
        X_val_selected = X_val_scaled[:, selected_indices]
        fold_feature_names = feature_names[selected_indices]
        features_after = X_train_selected.shape[1]
        
        print(f"[Fold {fold}] Features -> Before Lasso: {features_before} | After Lasso: {features_after}")
        
        # --- HYPERPARAMETER TUNING ---
        cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            estimator=model, param_grid=param_grids[model_name], 
            cv=cv_inner, scoring='roc_auc', n_jobs=2
        )
        grid_search.fit(X_train_selected, y_train)
        best_estimator = grid_search.best_estimator_
        optimal_params = str(grid_search.best_params_)
        
        print(f"[Fold {fold}] Optimal Parameters: {optimal_params}")
        
        y_pred = best_estimator.predict(X_val_selected)
        
        if hasattr(best_estimator, "predict_proba"):
            y_proba = best_estimator.predict_proba(X_val_selected)[:, 1]
        else:
            y_proba = best_estimator.decision_function(X_val_selected)
            
        model_confusion_matrices[model_name] += confusion_matrix(y_val, y_pred, labels=[0, 1])
        
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        roc_plotting_data[model_name]['fpr_list'].append(fpr)
        roc_plotting_data[model_name]['tpr_list'].append(tpr)
        
        # Metrics Calculations
        f_auc = roc_auc_score(y_val, y_proba)
        f_sens = recall_score(y_val, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
        f_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        f_prec = precision_score(y_val, y_pred, zero_division=0)
        f_f1 = f1_score(y_val, y_pred, zero_division=0)
        
        fold_aucs.append(f_auc)
        
        # --- LOG COMPREHENSIVE FOLD METRICS & TUNING INFO ---
        performance_records.append({
            'Model': model_name, 
            'Fold': fold, 
            'Features_Before_Lasso': features_before,
            'Features_After_Lasso': features_after,
            'Optimal_Parameters': optimal_params,
            'AUROC': f_auc,
            'Sensitivity': f_sens, 
            'Specificity': f_spec,
            'Precision': f_prec, 
            'F1_Score': f_f1
        })
        
        # Log Feature Importances
        importances = None
        if hasattr(best_estimator, 'feature_importances_'):
            importances = best_estimator.feature_importances_
        elif hasattr(best_estimator, 'coef_'):
            importances = np.abs(best_estimator.coef_[0])
            
        if importances is not None:
            for name_feat, score in zip(fold_feature_names, importances):
                importance_records.append({
                    'Model': model_name, 'Fold': fold,
                    'Feature': name_feat, 'Importance_Score': score
                })

    model_performance_summary[model_name] = np.mean(fold_aucs)

# =====================================================================
# 4. EXPORT EXTENSIVE METRIC LOGS TO CSV
# =====================================================================
df_perf = pd.DataFrame(performance_records)
df_perf.to_csv('detailed_fold_performance.csv', index=False)
print("Saved detailed fold metrics & tuning shapes to 'detailed_fold_performance.csv'")

if importance_records:
    df_imp = pd.DataFrame(importance_records)
    df_imp.to_csv('detailed_feature_importances.csv', index=False)
    print("Saved raw feature importances per fold to 'detailed_feature_importances.csv'")

# =====================================================================
# 5. PRINT AGGREGATED CONFUSION MATRICES
# =====================================================================
print("\n" + "="*50 + "\n   AGGREGATED CONFUSION MATRICES\n" + "="*50)
for model_name, cm in model_confusion_matrices.items():
    print(f"\nModel: {model_name}")
    print(f"True Negatives: {int(cm[0,0]):4d} | False Positives: {int(cm[0,1]):4d}")
    print(f"False Negatives: {int(cm[1,0]):4d} | True Positives:  {int(cm[1,1]):4d}")

# =====================================================================
# 6. EXPORT UNIFORM ROC COORDINATES TO CSV & PLOT
# =====================================================================
mean_fpr = np.linspace(0, 1, 100)
csv_export_dict = {'False_Positive_Rate': mean_fpr}

plt.figure(figsize=(10, 7))
for model_name, data in roc_plotting_data.items():
    tprs = []
    for fpr, tpr in zip(data['fpr_list'], data['tpr_list']):
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    
    csv_export_dict[f'{model_name.replace(" ", "_")}_True_Positive_Rate'] = mean_tpr
    plt.plot(mean_fpr, mean_tpr, label=f"{model_name} (AUC = {model_performance_summary[model_name]:.2f})", lw=2)

df_roc_export = pd.DataFrame(csv_export_dict)
df_roc_export.to_csv('roc_curves_data.csv', index=False)
print("Successfully exported ROC coordinates to 'roc_curves_data.csv'\n")

plt.plot([0, 1], [0, 1], linestyle='--', color='red', alpha=0.5)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curves (Cross-Validated Mean)')
plt.legend(loc="lower right")
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# =====================================================================
# 7. OPTION A: TRAIN AND EXPORT FINALIZED MODELS FOR ALL ALGORITHMS
# =====================================================================
print("\n" + "="*70)
print("FINAL COMPARATIVE PERFORMANCE LEADERBOARD (Mean AUROC)")
print("="*70)
sorted_leaderboard = sorted(model_performance_summary.items(), key=lambda x: x[1], reverse=True)
for rank, (name, score) in enumerate(sorted_leaderboard, 1):
    print(f" Rank {rank}: {name:<25} | Mean AUROC: {score:.4f}")
print("="*70 + "\n")

print("⚡ Initializing Final Whole-Dataset Training Phase for ALL Architectures...\n")

# Scale the full dataset for final training
full_scaler = StandardScaler()
X_full_scaled = full_scaler.fit_transform(X_matrix)

# Run a final Lasso model on 100% of the data to get the absolute consensus features
print("Running global Lasso selection to determine final consensus features...")
full_lasso_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
full_lasso = LassoCV(cv=full_lasso_cv, random_state=42, max_iter=5000)
full_lasso.fit(X_full_scaled, y)

full_features = np.where(full_lasso.coef_ != 0)[0]
if len(full_features) == 0: 
    full_features = np.arange(n_features)
    print("Warning: Global Lasso zeroed out all features. Defaulting to all features.")

X_full_selected = X_full_scaled[:, full_features]

print(f"Consensus Feature Footprint Selected: {len(full_features)} / {n_features} variations.\n")

# Save global scaling and selection assets
joblib.dump(full_scaler, 'production_scaler_global.pkl')
np.save('production_features_consensus.npy', full_features)

# Train, optimize, and save every single model architecture
for model_name, model in models.items():
    print(f"Finalizing architecture: {model_name}...")
    
    # Final global tuning pass across 100% of the pruned matrix
    final_grid = GridSearchCV(
        estimator=model, 
        param_grid=param_grids[model_name], 
        cv=5, 
        scoring='roc_auc', 
        n_jobs=-1
    )
    final_grid.fit(X_full_selected, y)
    
    best_production_model = final_grid.best_estimator_
    print(f"   -> Selected Parameters: {final_grid.best_params_}")
    
    # Save the finalized model file out to disk
    clean_name = model_name.replace(" ", "_")
    model_filename = f'final_model_{clean_name}.pkl'
    joblib.dump(best_production_model, model_filename)
    print(f"Successfully exported to '{model_filename}'\n")
