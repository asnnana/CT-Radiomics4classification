import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.interpolate import interp1d
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


train_data = pd.read_csv("./top_7_features_data.csv")  # loading data
X_train = train_data.drop(columns=["label"])
y_train = train_data["label"]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

test_data = pd.read_csv("./test_7_feature.csv")
X_test = test_data.drop(columns=["label"])
y_test = test_data["label"]
scaler = StandardScaler()
X_test = scaler.transform(X_test)

smote = SMOTE(random_state=42)  # over-sampling
X_train, y_train = smote.fit_resample(X_train, y_train)

classifier = SVC(kernel='linear')  # create classifier, SVC as an example
# classifier_rbf = SVC(kernel='rbf')
# classifier_dt = DecisionTreeClassifier(max_depth=5) 
# classifier_rf = RandomForestClassifier(n_estimators=100)

param_grid = {'C': [0.1, 1, 10, 100]}  # using GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_C = grid_search.best_params_['C']
classifier = SVC(kernel='linear', C=best_C)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)  # using model for prediction
y_pred_proba = classifier.decision_function(X_test) 

accuracy = accuracy_score(y_test, y_pred)
print("ACC:", accuracy)

auc = roc_auc_score(y_test, y_pred_proba) 
print("AUC:", auc)

for i, score in enumerate(y_pred_proba):
    print(f"data {i+1}: gets score {score}")

y_train_pred_proba = classifier.decision_function(X_train)  
for i, score in enumerate(y_train_pred_proba):
    print(f"data {i+1}: gets score {score}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("confusion matrix:\n", conf_matrix)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)  # for ROC curve
unique_fpr, unique_indices = np.unique(fpr, return_index=True)
unique_tpr = tpr[unique_indices]
interp_f = interp1d(unique_fpr, unique_tpr, kind='cubic')
smooth_fpr = np.linspace(0, 1, 100)
smooth_tpr = interp_f(smooth_fpr)
plt.figure(figsize=(8, 6))
plt.plot(smooth_fpr, smooth_tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Smoothing Receiver Operating Characteristic (ROC) Curve')
plt.legend()
# plt.show()