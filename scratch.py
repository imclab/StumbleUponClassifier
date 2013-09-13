import numpy as np
from scipy.sparse import coo_matrix, vstack


def get_vector():
    featureVector = np.zeros(14)
    featureVector[1] = 1
    return featureVector


def main():
    l = []
    current = get_vector()
    for i in range(10):
        l.append(get_vector())
        current = np.vstack([current, get_vector()])

    m = np.matrix(l)
    print type(m)
    print m[0]
    print type(m[0])

    print current
    print type(current)


if __name__ == '__main__':
    main()