"""
The score to beat:
20 Fold CV Score:  0.876809206097
"""
from __future__ import division
import sys
import os
import numpy as np
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
import pandas as pd
from sklearn.svm import SVC
from scipy import sparse


def get_train_data(stem):
    if stem:
        return [line.strip() for line in open('train.tsv.clean.stem')]
    else:
        return list(np.array(pd.read_table('train.tsv'))[:, 2])


def get_test_data(stem):
    if stem:
        return [line.strip() for line in open('test.tsv.clean.stem')]
    else:
        return list(np.array(pd.read_table('test.tsv'))[:, 2])


def stack_tfidf_features(X):
    pass


def get_tfidf_features(X_all):
    print 'fitting vectorizer'
    tfidfVectorizer = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word',
                                      token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=1, smooth_idf=1,
                                      sublinear_tf=1)
    tfidfVectorizer.fit(X_all)
    return tfidfVectorizer.transform(X_all)


def stack_topic_features(X):
    print 'stacking topic features'
    X_all_topics = np.genfromtxt("features.csv", delimiter=",")
    X_all = sparse.hstack((X, X_all_topics))
    return X_all.tocsr()


def stack_category_features(X):
    print 'stacking category features'
    X_all_cats = np.genfromtxt("cat_vectors.csv", delimiter=",")
    X_all = sparse.hstack((X, X_all_cats))
    return X_all.tocsr()


def stack_is_news_features(X):
    print 'stacking is_news features'
    X_all_is_news = np.asmatrix(np.genfromtxt('is_news.csv', delimiter=',')).transpose()
    X_all = sparse.hstack((X, X_all_is_news))
    return X_all.tocsr()


def write_submission(prediction):
    testfile = pd.read_csv('test.tsv', sep="\t", na_values=['?'], index_col=1)
    prediction_df = pd.DataFrame(prediction, index=testfile.index, columns=['label'])
    prediction_df.to_csv('submission.csv')
    print "submission file created.."


def main():
    os.chdir(sys.argv[1])
    stem = True
    trainData = get_train_data(stem)
    testData = get_test_data(stem)
    y = np.array(pd.read_table('train.tsv'))[:, -1]

    rd = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                               C=1, fit_intercept=True, intercept_scaling=1.0,
                               class_weight=None, random_state=None)

    svc = SVC(probability=True)

    X_all = trainData + testData
    lenTrain = len(trainData)

    X_all = get_tfidf_features(X_all)
    X_all = stack_category_features(X_all)
    X_all = stack_topic_features(X_all)
    X_all = stack_is_news_features(X_all)

    # print "20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(
    #     rd, X_all[:lenTrain], y, cv=20, scoring='roc_auc'
    # ))
    print "20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(
        svc, X_all[:lenTrain], y, cv=20, scoring='roc_auc'
    )), 'SVM'

    # rd.fit(X_all[:lenTrain], y)
    # predictions = rd.predict_proba(X_all[lenTrain:])[:, 1]
    # write_submission(predictions)


if __name__ == '__main__':
    main()