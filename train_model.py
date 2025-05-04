
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

df = pd.read_csv('creditcard.csv')

# Take all frauds (class = 1) and equal number of non-frauds (class = 0)
fraud = df[df['Class'] == 1]
non_fraud = df[df['Class'] == 0].sample(n=len(fraud), random_state=42)

df_small = pd.concat([fraud, non_fraud]).sample(frac=1, random_state=42)

X = df_small[['V10','V12','V14','V16','V17','Amount']]
y = df_small['Class']


# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models with balanced settings
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_scaled, y)

rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_scaled, y)

svm = SVC(probability=True, class_weight='balanced')
svm.fit(X_scaled, y)

# For XGBoost, set scale_pos_weight = (negatives / positives)
pos = sum(y == 1)
neg = sum(y == 0)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                    scale_pos_weight=(neg/pos), random_state=42)
xgb.fit(X_scaled, y)

# Save scaler and models to 'models/' directory
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(lr, 'models/model_lr.pkl')
joblib.dump(rf, 'models/model_rf.pkl')
joblib.dump(svm, 'models/model_svm.pkl')
joblib.dump(xgb, 'models/model_xgb.pkl')
print("Models trained and saved successfully.")