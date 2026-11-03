from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import joblib
from IPython.display import display
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             classification_report, ConfusionMatrixDisplay, make_scorer)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

SEED = 42
np.random.seed(SEED)
sns.set_theme(style='whitegrid')
pd.set_option('display.max_columns', 25)
print('Python:', sys.version.split()[0], '| scikit-learn:', sklearn.__version__)
train_raw = pd.read_csv('Train2021.csv')
test_raw = pd.read_csv('Test Data.csv')

RESULTS_DIR = Path('heatguard_results')
TARGET = 'Thermal sensation'
CLASSES = ['neutral',
           'slightly warm',
           'warm',
           'hot']

FEATURES = ['Temperature',
            'Humidity',
            'hrv_mean_hr',
            'hrv_rmssd',
            'hrv_sdnn',
            'hrv_lf_hf_ratio']

def load_data(path):
    if not path.is_file():
        raise FileNotFoundError(f'Update the CSV path: {path}')
    df = pd.read_csv(path)
    required = set(FEATURES + [TARGET, 'subject', 'experiment'])
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {required - set(df.columns)}')
    for col in ['subject', 'experiment', TARGET]:
        if df[col].isna().any():
            raise ValueError(f'Missing grouping/target values in {col}')
        df[col] = df[col].astype(str).str.strip()
        if df[col].eq('').any():
            raise ValueError(f'Empty values in {col}')
    if not df[TARGET].isin(CLASSES).all():
        raise ValueError('Unexpected thermal-sensation labels')
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return df

overlap = set(train_raw.subject) & set(test_raw.subject)
assert not overlap, f'Overlapping participant names: {overlap}'
display(pd.DataFrame({'file': ['Train2021', 'Test Data'],
    'rows': [len(train_raw), len(test_raw)],
    'columns': [len(train_raw.columns), len(test_raw.columns)],
    'participants': [train_raw.subject.nunique(), test_raw.subject.nunique()]}))
display(train_raw[FEATURES + [TARGET]].head())
def summarize_sessions(df):
    keys = ['subject', 'experiment']
    counts = df.groupby(keys)[TARGET].nunique()
    assert counts.eq(1).all(), 'Mixed labels: revise the session definition.'
    return df.groupby(keys, as_index=False).agg(
        **{c: (c, 'median') for c in FEATURES},
        actual=(TARGET, 'first'), n_rows=(TARGET, 'size'))

train = summarize_sessions(train_raw)
test = summarize_sessions(test_raw)
X_train = train[FEATURES].copy()
y_train = train.actual.copy()
groups = train.subject.copy()
X_test = test[FEATURES].copy()
y_test = test.actual.copy()
assert set(y_train) == set(CLASSES)
assert not X_train.isna().all().any(), 'At least one feature is entirely missing.'
print('Training groups:', len(train), '| Test groups:', len(test))
print('Training participants:', groups.nunique())
display(X_train.isna().sum().to_frame('missing_training_session_values'))
display(y_train.value_counts().reindex(CLASSES).to_frame('training_sessions'))


fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.countplot(data=train, x='actual', order=CLASSES, ax=axes[0], color='steelblue')
axes[0].tick_params(axis='x', rotation=25)
axes[0].set_title('Training class balance (sessions)')
sns.scatterplot(data=train, x='Temperature', y='Humidity', hue='actual', ax=axes[1])
axes[1].set_title('Limited environmental coverage')
sns.boxplot(data=train, x='actual', y='hrv_mean_hr', order=CLASSES, ax=axes[2])
axes[2].tick_params(axis='x', rotation=25)
axes[2].set_title('Mean heart rate by recorded label')
plt.tight_layout()
plt.show()
display(train_raw.groupby('experiment')[TARGET].agg(['nunique', 'first']))
display(train_raw.groupby(['Temperature', 'Humidity', TARGET]).size().to_frame('rows'))

def make_pipeline(estimator):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', clone(estimator)),
    ])

cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
folds = list(cv.split(X_train, y_train, groups))
for i, (fit_idx, val_idx) in enumerate(folds, start=1):
    assert not set(groups.iloc[fit_idx]) & set(groups.iloc[val_idx])
    assert set(y_train.iloc[fit_idx]) == set(CLASSES), 'Training fold lacks a class.'
    assert set(y_train.iloc[val_idx]) == set(CLASSES), 'Validation fold lacks a class.'
    print(f'Fold {i}: {groups.iloc[fit_idx].nunique()} training participants, '
          f'{groups.iloc[val_idx].nunique()} validation participants')