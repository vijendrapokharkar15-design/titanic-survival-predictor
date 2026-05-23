import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
import os

# load data
df = pd.read_csv("titanic.csv")

# basic cleaning
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna("S")
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# feature engineering — same as day 22
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["FareLog"] = np.log1p(df["Fare"])
df["HasCabin"] = df["Cabin"].notna().astype(int)
df["Sex_encoded"] = (df["Sex"] == "female").astype(int)

# title extraction — this was the big win in day 22
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
title_map = {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master"}
df["Title"] = df["Title"].map(title_map).fillna("Other")
for t in ["Mr", "Mrs", "Miss", "Master"]:
    df[f"Title_{t}"] = (df["Title"] == t).astype(int)

features = [
    "Pclass",
    "Age",
    "FareLog",
    "FamilySize",
    "IsAlone",
    "HasCabin",
    "Sex_encoded",
    "Title_Mr",
    "Title_Mrs",
    "Title_Miss",
    "Title_Master",
]

X = df[features]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# random forest — best model from day 22-25
rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=4, random_state=42)
rf.fit(X_train, y_train)

# quick check
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="roc_auc")
test_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
test_acc = accuracy_score(y_test, rf.predict(X_test))

print(f"CV AUC:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test Acc: {test_acc:.4f}")
print()
print(
    classification_report(y_test, rf.predict(X_test), target_names=["Died", "Survived"])
)

# save model and feature list
os.makedirs("model", exist_ok=True)
with open("model/rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)
with open("model/features.pkl", "wb") as f:
    pickle.dump(features, f)

print("model saved to model/rf_model.pkl ✅")
