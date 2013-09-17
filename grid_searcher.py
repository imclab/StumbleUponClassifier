from __future__ import print_function

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse


traindata = list(np.array(pd.read_table('/media/Storage/workspace/sudata/train.tsv'))[:, 2])
testdata = list(np.array(pd.read_table('/media/Storage/workspace/sudata/test.tsv'))[:, 2])
X_all_topics = np.genfromtxt("/media/Storage/workspace/sudata/features.csv", delimiter=",")
X_all_cats = np.genfromtxt("/media/Storage/workspace/sudata/cat_vectors.csv", delimiter=",")



y = np.array(pd.read_table('/media/Storage/workspace/sudata/y'))[:, 0]

tfidfVectorizer = TfidfVectorizer(min_df=3,    max_features=None, strip_accents='unicode',
            analyzer='word', token_pattern=r'\w{1,}',
            ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1)

lentrain = len(traindata)

X_all = traindata + testdata
tfidfVectorizer.fit(X_all)
X_all = tfidfVectorizer.transform(X_all)

X_all = sparse.hstack((X_all, X_all_cats))
X_all = X_all.tocsr()
X_all = sparse.hstack((X_all, X_all_topics))
X_all = X_all.tocsr()

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(X_all[:lentrain], y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters_svc = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

tuned_parameters_lr = [{}]

scores = ['roc_auc']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1, probability=True), tuned_parameters_svc, cv=20, scoring=score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()