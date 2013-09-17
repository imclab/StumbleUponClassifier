"""
beating the benchmark @StumbleUpon Evergreen Challenge
__author__ : Abhishek Thakur
"""

# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
import pandas as p
from nltk import word_tokenize
from nltk import PorterStemmer
import csv
import json


def stem_and_filter_dataset(dataset, outFileName):
    stemmer = PorterStemmer()
    stopList = [line.strip() for line in open('../StumbleUponData/stoplist.txt')]
    outFile = open(outFileName, 'wb')
    out = csv.writer(outFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    counter = 0;
    for document in dataset:
        counter += 1
        if counter % 500 == 0:
            print counter
        document = json.loads(document)
        try:
            title = document['title']
        except KeyError:
            title = ''
        body = document['body']
        if title is None:
            title = ''
        if body is None:
            body = ''
        title = title.encode('ascii', 'ignore')
        body = body.encode('ascii', 'ignore')
        content = title + ' ' + body
        content = content.lower()
        tokens = word_tokenize(content)
        # tokens = [stemmer.stem(t) for t in tokens if t not in stopList]
        tokens = [w for w in tokens if w not in stopList]
        out.writerow([' '.join(tokens)])
    outFile.close()


def clean_is_news():
    outFile = open('../StumbleUponData/is_news', 'wb')
    out = csv.writer(outFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)


def main():
    clean_is_news()
    exit(0)
    print "loading data.."
    traindata = [line.strip() for line in open('../StumbleUponData/train.tsv.clean.stem')]
    testdata = [line.strip() for line in open('../StumbleUponData/test.tsv.clean.stem')]
    y = np.array(p.read_table('../StumbleUponData/train.tsv'))[:,-1]


    tfv = TfidfVectorizer(
            min_df=3,
            max_features=None,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 2),
            use_idf=1,
            smooth_idf=1,
            sublinear_tf=1
    )

    rd = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                               C=1, fit_intercept=True, intercept_scaling=1.0,
                               class_weight=None, random_state=None)

    X_all = traindata + testdata
    lentrain = len(traindata)

    print "fitting pipeline"
    tfv.fit(X_all)
    print "transforming data"
    X_all = tfv.transform(X_all)

    X = X_all[:lentrain]
    X_test = X_all[lentrain:]

    print "20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(rd, X, y, cv=20, scoring='roc_auc'))

    print "training on full data"
    rd.fit(X,y)
    pred = rd.predict_proba(X_test)[:,1]
    testfile = p.read_csv('../StumbleUponData/test.tsv', sep="\t", na_values=['?'], index_col=1)
    pred_df = p.DataFrame(pred, index=testfile.index, columns=['label'])
    pred_df.to_csv('../StumbleUponData/original_benchmark.csv')
    print "submission file created.."

if __name__=="__main__":
    main()
