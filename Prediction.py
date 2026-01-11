import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('cervical_cancer.csv')

print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

df = df.replace('?', np.nan)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nMissing values after conversion:")
print(df.isnull().sum())

target_col = 'Biopsy'

if target_col in df.columns:
    X = df.drop([target_col], axis=1)
    y = df[target_col]
    
    other_targets = ['Schiller', 'Citology', 'Hinselmann']
    for t in other_targets:
        if t in X.columns:
            X = X.drop([t], axis=1)
else:
    print(f"Target column '{target_col}' not found. Available columns: {df.columns.tolist()}")
    exit()

print(f"\nTarget variable distribution:")
print(y.value_counts())
print(f"Positive cases: {(y==1).sum()} ({(y==1).sum()/len(y)*100:.2f}%)")

imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

print("\n" + "="*70)
print("MODEL EVALUATION RESULTS")
print("="*70)

for name, model in models.items():
    print(f"\n{name}:")
    print("-" * 50)
    
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    print(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'roc_auc': roc_auc
    }

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

for idx, (name, result) in enumerate(results.items()):
    if idx < 3:
        row = idx // 2
        col = idx % 2
        cm = confusion_matrix(y_test, result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col])
        axes[row, col].set_title(f'Confusion Matrix - {name}')
        axes[row, col].set_ylabel('Actual')
        axes[row, col].set_xlabel('Predicted')

ax = axes[1, 1]
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
    ax.plot(fpr, tpr, label=f"{name} (AUC = {result['roc_auc']:.3f})")

ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
print("\n\nVisualization saved as 'model_evaluation.png'")

rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("="*70)
print("TOP 10 MOST IMPORTANT FEATURES")
print("="*70)
print(feature_importance.head(10))

plt.figure(figsize=(10, 6))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importance (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\nFeature importance plot saved as 'feature_importance.png'")

best_model_name = max(results.items(), key=lambda x: x[1]['roc_auc'])[0]
best_model = results[best_model_name]['model']

print(f"\n\nBest performing model: {best_model_name}")
print(f"ROC-AUC Score: {results[best_model_name]['roc_auc']:.4f}")

def predict_cervical_cancer(features):
    if isinstance(features, dict):
        features = pd.DataFrame([features])
    
    features_imputed = imputer.transform(features)
    features_scaled = scaler.transform(features_imputed)
    
    prediction = best_model.predict(features_scaled)[0]
    probability = best_model.predict_proba(features_scaled)[0, 1]
    
    return prediction, probability

print("\n" + "="*70)
print("Model training complete! You can use predict_cervical_cancer() for new predictions.")
print("="*70)