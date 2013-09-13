"""

"""
import numpy as np
import pandas as pd


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


def main():
    traindata = list(np.array(pd.read_table('/media/Storage/workspace/sudata/train.tsv'))[:, 3])
    testdata = list(np.array(pd.read_table('/media/Storage/workspace/sudata/test.tsv'))[:, 3])
    alldata = traindata + testdata
    print len(alldata)
    current = alchemy_category_vectorizer(alldata[0])
    for example in alldata[1:]:
        current = np.vstack([current, alchemy_category_vectorizer(example)])
    print len(current)
    print current[0]
    print type(current)
    freature_df = pd.DataFrame(current, index=None)
    freature_df.to_csv('/media/Storage/workspace/sudata/cat_vectors.csv', header=False, index=False)

if __name__ == '__main__':
    main()