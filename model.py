import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pickle

# Load the dataset
df = pd.read_csv('C:\\Users\\ADITY BARNWAL\\OneDrive\\Desktop\\lung_cancer_application\\lung_cancer_app\\surveylungcancer3.csv')

# Map target values: "NO" = No Cancer → 0, "YES" = Cancer → 1
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({"NO": 0, "YES": 1})

# Split features and target
X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (for certain models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True),
    'XGBoost': XGBClassifier(eval_metric='logloss')
}

# Train, evaluate, and select best model
best_model = None
best_score = 0
best_model_name = ''
results = []

for name, model in models.items():
    if name in ['Logistic Regression', 'KNN', 'SVM']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    results.append((name, acc))
    print(f"{name} Accuracy: {acc:.4f}")
    
    if acc > best_score:
        best_score = acc
        best_model = model
        best_model_name = name

# Save best model as model.pkl
with open('model.pkl', 'wb') as f:#open - > read write append new data, with - > close file automattically to access it from anywhere, f(reference) - >is just a var name to the open file  object
    pickle.dump(best_model, f)

print(f"\nBest Model: {best_model_name} with Accuracy: {best_score:.4f}")
