"""
classifier.py
Train and evaluate RandomForest classifier and save/load model utilities.
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_random_forest(X, y, n_estimators=300, cv=5, seed=42, verbose=True):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    if cv and cv > 1:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        if verbose:
            print(f"[RF] CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    clf.fit(X, y)
    return clf

def evaluate(clf, X_test, y_test):
    ypred = clf.predict(X_test)
    acc = accuracy_score(y_test, ypred)
    print(f"[EVAL] Test accuracy: {acc:.4f}")
    print(classification_report(y_test, ypred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, ypred))
    return acc, ypred

def save_model(clf, path):
    joblib.dump(clf, path)

def load_model(path):
    return joblib.load(path)
 
