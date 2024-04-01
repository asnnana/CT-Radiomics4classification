import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("./train_feature3_4lasso.csv")  # loading data
X = data.drop(columns=["label"])
y = data["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

alpha = 0.01  # settings of lasso regressor
lasso = Lasso(alpha=alpha)
lasso.fit(X_scaled, y)

selected_features = X.columns[lasso.coef_ != 0]  # sorting data sccording to the correlation with y
correlations = []  
for feature in selected_features:
    correlation = X[feature].corr(y)
    correlations.append((feature, correlation))
correlations.sort(key=lambda x: abs(x[1]), reverse=True)
sorted_selected_features = [x[0] for x in correlations]

selected_data = data[sorted_selected_features]
selected_data["label"] = y  
selected_data.to_csv("selected_data.csv", index=False)

# top 7 features data 
top_7_features_data = data[["label"] + sorted_selected_features[:7]]  
top_7_features_data.to_csv("top_7_features_data.csv", index=False)

# top 7 features and correlation values
top_7_correlations_data = pd.DataFrame(correlations[:7], columns=["Feature", "Correlation"])
top_7_correlations_data.to_csv("top_7_correlations_data.csv", index=False)
