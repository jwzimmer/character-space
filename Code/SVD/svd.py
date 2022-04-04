import codecs
import json

import numpy as np
import pandas as pd
import scipy as sp
from scipy import linalg


def drop_columns_before_SVD(drop_columns, df):
    """ If columns need to be dropped from the dataframe
    before SVD is run, pass them as a list. Returns a new
    dataframe. """
    new_df = df
    for column in drop_columns:
        new_df = new_df.drop(column, axis=1)
    return new_df


def subtract_mean(df):
    df_mean = df.mean()
    new_df = df - df_mean
    return new_df


def run_SVD(df):
    """ df is the dataframe you want to run SVD on;
    this function is mostly copy-pasted from https://stackoverflow.com/questions/55777664/getting-singular-values-of-numpy-data-columns-in-order/55785021
    """
    X = df.to_numpy()
    # decompose
    U, D, V = np.linalg.svd(X)
    # get dim of X
    M, N = X.shape
    # Construct sigma matrix in SVD (it simply adds null row vectors to match the dim of X)
    Sig = sp.linalg.diagsvd(D, M, N)
    # Now you can get X back:
    remakeX = np.dot(U, np.dot(Sig, V))
    assert np.sum(remakeX - X) < 0.00001
    return df, U, D, V, Sig, X, remakeX


def write_array_to_json(numpy_array, filepath):
    output = numpy_array.tolist()
    json.dump(output, codecs.open(filepath, 'w', encoding='utf-8'),
              separators=(',', ':'),
              sort_keys=True,
              indent=4)  ### this saves the array in .json format
    return None


def write_array_to_npy(numpy_array, name):
    np.save(name, numpy_array)
    return None


def write_df_to_json(df, filepath):
    df.to_json(filepath)
    return None


if __name__ == '__main__':
    path_to_df_of_data = "lsa.json"
    data_df = pd.read_json(path_to_df_of_data)

    data_df = subtract_mean(data_df)
    df, U, D, V, Sig, X, remakeX = run_SVD(data_df)
    write_array_to_npy(U, "U.npy")
    write_array_to_npy(D, "D.npy")
    write_array_to_npy(V, "V.npy")
