"""

"""
import numpy as np
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
import pandas as pd
from sklearn.svm import SVC
from scipy import sparse


def main():
    # traindata = list(np.array(pd.read_table('/home/gavin/dev/StumbleUponData/train.tsv'))[:, 2])
    traindata = list(np.array(pd.read_table('/media/Storage/workspace/sudata/train.tsv'))[:, 2])

    # testdata = list(np.array(pd.read_table('/home/gavin/dev/StumbleUponData/test.tsv'))[:, 2])
    testdata = list(np.array(pd.read_table('/media/Storage/workspace/sudata/test.tsv'))[:, 2])

    # X_all_topics = np.genfromtxt("/home/gavin/dev/StumbleUponData/features.csv", delimiter=",")
    X_all_topics = np.genfromtxt("/media/Storage/workspace/sudata/features.csv", delimiter=",")

    X_all_cats = np.genfromtxt("/media/Storage/workspace/sudata/cat_vectors.csv", delimiter=",")

    # y = np.array(pd.read_table('/home/gavin/dev/StumbleUponData/train.tsv'))[:, -1]
    y = np.array(pd.read_table('/media/Storage/workspace/sudata/y'))[:, 0]

    tfidfVectorizer = TfidfVectorizer(min_df=3,    max_features=None, strip_accents='unicode',
                analyzer='word', token_pattern=r'\w{1,}',
                ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1)

    rd = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                               C=1, fit_intercept=True, intercept_scaling=1.0,
                               class_weight=None, random_state=None)

    print 'fitting vectorizer'
    # tfidfVectorizer.fit(testdata)
    # X_test = tfidfVectorizer.transform(testdata)

    X_all = traindata + testdata
    lentrain = len(traindata)
    tfidfVectorizer.fit(X_all)
    X_all = tfidfVectorizer.transform(X_all)

    X_all = sparse.hstack((X_all, X_all_cats))
    X_all = X_all.tocsr()

    X_all = sparse.hstack((X_all, X_all_topics))
    X_all = X_all.tocsr()

    print "RD 20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(rd, X_all[:lentrain], y, cv=20, scoring='roc_auc'))

    # X_test = X_all[lentrain:]
    # X = X_all[:lentrain]

    # X = X_all_topics[:lentrain]
    # X_test = X_all_topics[lentrain:]
    # print "RD 20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(rd, X, y, cv=20, scoring='roc_auc'))

    rd.fit(X_all[:lentrain], y)
    pred = rd.predict_proba(X_all[lentrain:])[:, 1]

    # testfile = pd.read_csv('/home/gavin/dev/StumbleUponData/test.tsv', sep="\t", na_values=['?'], index_col=1)
    testfile = pd.read_csv('/media/Storage/workspace/sudata/test.tsv', sep="\t", na_values=['?'], index_col=1)
    pred_df = pd.DataFrame(pred, index=testfile.index, columns=['label'])
    # pred_df.to_csv('/home/gavin/dev/StumbleUponData/submission.csv')
    pred_df.to_csv('/media/Storage/workspace/sudata/submission.csv')
    print "submission file created.."


if __name__ == '__main__':
    main()