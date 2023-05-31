#######################################
########---Importing Libraries---######
#######################################
import joblib
import pickle
import warnings
import graphviz
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msn
from sklearn.svm import SVC
from skompiler import skompile
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, confusion_matrix


warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)



#######################################
##--Data Reading and Understanding---##
#######################################
df = pd.read_csv(r"/Users/abdullahorzan/Desktop/Miuul/DSML 11 Datavaders/final project/Datasets/1200_song_mapped.csv")

def check_df(dataframe, head=5):
    print("############### SHAPE ###############")
    print(dataframe.shape)
    print("############### TYPES ###############")
    print(dataframe.dtypes)
    print("############### HEAD ###############")
    print(dataframe.head(head))
    print("############### TAIL ###############")
    print(dataframe.tail(head))
    print("############### NULL ###############")
    print(dataframe.isnull().sum())
    print("############### QUANTILES ###############")
    print(dataframe.describe().T)

check_df(df)


## Visualizations

plt.pie(df["labels"].value_counts().values, labels = df["labels"].value_counts().index, autopct='%1.1f%%')
plt.legend(title="Emotions")
plt.show(block=True)

sns.countplot(df, x="labels")
plt.show(block=True)


## Missing Values
msn.matrix(df, color=(0.1, 0.1, 0.1))
plt.show(block=True)


#######################################
########---Data PreProcessing---#######
#######################################

## Shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)

## Columns to drop
df.drop(["Unnamed: 0", "popularity", "time signature"], inplace=True, axis=1)

## Adjust column names
df.columns = df.columns.str.lower()


#######################################
########---Feature Engineering---######
#######################################

## Feature Extraction
df["spec_rate"] = df["speechiness"] / df["duration (ms)"]

# Variables for model input

X = df.drop(["track", "artist", "uri", "labels", "key"], axis=1)
y = df["labels"]

## Scaling
rb = RobustScaler()
X = rb.fit_transform(X)

## Export Scaler
with open('rb_scaler.pkl', 'wb') as file:
    pickle.dump(rb, file)

#######################################
############---Modelling---############
#######################################

## Base Model
def base_models(X, y):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier())
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=10)
        print(f"{round(cv_results['test_score'].mean(), 4)} ({name}) ")

base_models(X, y)

## LightGBM with cross validate
light_model = LGBMClassifier(random_state=44)
cv_results = cross_validate(light_model, X, y, cv=10, scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"])
cv_results['test_accuracy'].mean()
cv_results['test_f1_macro'].mean()
cv_results['test_precision_macro'].mean()
cv_results['test_recall_macro'].mean()

## Hyperparameter optimization
lgbm_params = {"learning_rate": [0.01, 1.0],
               'num_leaves': [24, 80],
               "n_estimators": [950, 1000, 1050],
               "colsample_bytree": [0.3, 0.4, 0.6, 0.7],
               'max_depth': [5, 30],
               'subsample': [0.01, 1.0]}


lgbm_best_grid = GridSearchCV(light_model, lgbm_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)
lgbm_best_grid.best_params_
lgbm_final = light_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)
cv_results = cross_validate(lgbm_final, X, y, cv=10, scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"])
cv_results['test_accuracy'].mean()
cv_results['test_f1_macro'].mean()
cv_results['test_precision_macro'].mean()
cv_results['test_recall_macro'].mean()

## Export model
joblib.dump(lgbm_final, 'lgbm_final.pkl')


## Plot Feature Importance
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)

plot_importance(lgbm_final, X)


