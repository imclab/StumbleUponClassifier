"""

"""

# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
import pandas as p
from sklearn import svm
from scipy import sparse


loadData = lambda f: np.genfromtxt(open(f, 'r'), delimiter=' ')


def main():
    print "loading data..."
    # load cleaned test and a/b test with default file
    traindata = list(np.array(p.read_table('../StumbleUponData/train.tsv'))[:, 2])
    testdata = list(np.array(p.read_table('../StumbleUponData/test.tsv'))[:, 2])
    trainAlchemyCats = list(np.array(p.read_table('../StumbleUponData/train.tsv'))[:, 3])
    testAlchemyCats = list(np.array(p.read_table('../StumbleUponData/test.tsv'))[:, 3])
    allAlchemyCats = trainAlchemyCats + testAlchemyCats
    y = np.array(p.read_table('../StumbleUponData/train.tsv'))[:, -1]

    tfidfVectorizer = TfidfVectorizer(
        min_df=3, max_features=None, strip_accents='unicode',
        analyzer='word', token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1)

    rd = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                               C=1, fit_intercept=True, intercept_scaling=1.0,
                               class_weight=None, random_state=None)

    # clf = svm.SVC(probability=True)

    X_all = traindata + testdata
    lentrain = len(traindata)

    print "fitting pipeline"
    # tfidfVectorizer.fit(X_all)
    print "transforming data"
    # X_all = tfidfVectorizer.transform(X_all)

    alchemy_vector = alchemy_category_vectorizer(allAlchemyCats[0])
    print len(X_all), len(alchemy_vector)
    X_all = sparse.hstack((alchemy_vector, X_all)).tocsr()

    # for index, X in enumerate(X_all[1:]):
    #     alchemy_vector = alchemy_category_vectorizer(allAlchemyCats[index])
    #     sparse.hstack((alchemy_vector, X)).tocsr()
        # row = sparse.hstack((alchemy_vector, X)).tocsr()
        # X_stacked = sparse.vstack((X_stacked, row)).tocsr()

    X = X_all[:lentrain]
    X_test = X_all[lentrain:]
    # X = X_stacked[:lentrain]
    # X_test = X_stacked[lentrain:]

    print "20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(rd, X, y, cv=20, scoring='roc_auc'))
    # print "20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(clf, X, y, cv=20, scoring='roc_auc'))

    print "training on full data"
    rd.fit(X, y)
    # clf.fit(X, y)

    pred = rd.predict_proba(X_test)[:, 1]
    # pred = clf.predict(X_test)[:, 1]

    testfile = p.read_csv('../StumbleUponData/test.tsv', sep="\t", na_values=['?'], index_col=1)
    pred_df = p.DataFrame(pred, index=testfile.index, columns=['label'])
    pred_df.to_csv('../StumbleUponData/benchmark.csv')
    print "submission file created.."


def alchemy_category_vectorizer(alchemy_category):
    featureVector = np.zeros(14)
    if alchemy_category == '?':
        featureVector[0] = 1
    elif alchemy_category == 'recreation':
        featureVector[1] = 1
    elif alchemy_category == 'arts_entertainment':
        featureVector[2] = 1
    elif alchemy_category == 'business':
        featureVector[3] = 1
    elif alchemy_category == 'health':
        featureVector[4] = 1
    elif alchemy_category == 'sports':
        featureVector[5] = 1
    elif alchemy_category == 'culture_politics':
        featureVector[6] = 1
    elif alchemy_category == 'computer_internet':
        featureVector[7] = 1
    elif alchemy_category == 'science_technology':
        featureVector[8] = 1
    elif alchemy_category == 'gaming':
        featureVector[9] = 1
    elif alchemy_category == 'religion':
        featureVector[10] = 1
    elif alchemy_category == 'law_crime':
        featureVector[11] = 1
    elif alchemy_category == 'unknown':
        featureVector[12] = 1
    elif alchemy_category == 'weather':
        featureVector[13] = 1
    return featureVector


if __name__=="__main__":
    main()