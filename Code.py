import numpy as np
import pandas as pd
import warnings
import optuna
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree  import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import itertools
import os

def le(df):
    for col in df.columns:
        if df[col].dtype == 'object':
                label_encoder = LabelEncoder()
                df[col] = label_encoder.fit_transform(df[col])
                
def objective(trial):
    n_neighbors = trial.suggest_int('KNN_n_neighbors', 2, 16, log=False)
    classifier_obj = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier_obj.fit(x_train, y_train)
    accuracy = classifier_obj.score(x_test, y_test)
    return accuracy

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train=pd.read_csv('Train_data.csv')
test=pd.read_csv('Test_data.csv')

train.info()
train.head()
train.describe()
train.describe(include='object')
total = train.shape[0]
missing_columns = [col for col in train.columns if train[col].isnull().sum() > 0]
for col in missing_columns:
    null_count = train[col].isnull().sum()
    per = (null_count/total) * 100
    print(f"{col}: {null_count} ({round(per, 3)}%)")
    
le(train)
le(test)

train.drop(['num_outbound_cmds'], axis=1, inplace=True)
test.drop(['num_outbound_cmds'], axis=1, inplace=True)
train.head()

X_train = train.drop(['class'], axis=1)
Y_train = train['class']

rfc = RandomForestClassifier()

rfe = RFE(rfc, n_features_to_select=10)
rfe = rfe.fit(X_train, Y_train)

feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), X_train.columns)]
selected_features = [v for i, v in feature_map if i==True]

selected_features

X_train = X_train[selected_features]

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
test = scale.fit_transform(test)

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, train_size=0.70, random_state=2)

print()
print("-----------KNN Algorithm-----------")

study_KNN = optuna.create_study(direction='maximize')
study_KNN.optimize(objective, n_trials=1)
KNN_model = KNeighborsClassifier(n_neighbors=study_KNN.best_trial.params['KNN_n_neighbors'])
KNN_model.fit(x_train, y_train)

KNN_train, KNN_test = KNN_model.score(x_train, y_train), KNN_model.score(x_test, y_test)
y_pred_train = KNN_model.predict(x_train)
y_pred_test = KNN_model.predict(x_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Train Score: {KNN_train}")
print(f"Test Score: {KNN_test}")
print("-----------------------------------")
print()
print("-----------SVM Algoritham----------")
svm = SVC()
svm.fit(x_train, y_train)
svm_train, svm_test = svm.score(x_train, y_train), svm.score(x_test, y_test)
y_pred_train = svm.predict(x_train)
y_pred_test = svm.predict(x_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Train Score: {svm_train}")
print(f"Test Score: {svm_test}")
print("-----------------------------------")
print()
print("-----------Random Forest Algorithm-----------")
study_RF = optuna.create_study(direction='maximize')
study_RF.optimize(objective, n_trials=1)
RF_model = RandomForestClassifier()
RF_model.fit(x_train, y_train)
RF_train, RF_test = RF_model.score(x_train, y_train), RF_model.score(x_test, y_test)
y_pred_train = RF_model.predict(x_train)
y_pred_test = RF_model.predict(x_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Train Score: {RF_train}")
print(f"Test Score: {RF_test}")
print("-----------------------------------")
print()
print("-----------XGBoost Algorithm-----------")
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'random_state': 42,
        'objective': 'binary:logistic'
    }
    model = xgb.XGBClassifier(**params)
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    return accuracy
study_XGB = optuna.create_study(direction='maximize')
study_XGB.optimize(objective, n_trials=50)
XGB_best_params = study_XGB.best_params
XGB_model = xgb.XGBClassifier(**XGB_best_params)
XGB_model.fit(x_train, y_train)
XGB_train, XGB_test = XGB_model.score(x_train, y_train), XGB_model.score(x_test, y_test)

y_pred_train = XGB_model.predict(x_train)
y_pred_test = XGB_model.predict(x_test)

accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Train Score: {XGB_train}")
print(f"Test Score: {XGB_test}")
print("-----------------------------------")
print()
print("-----------Decision Tree-----------")
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.3f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("-----------------------------------")