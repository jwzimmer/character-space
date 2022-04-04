import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def vector_barchart(vector_names, vector, n, style="by_mag", ascending=False):
    """ vector_names should be the labels for the values in the vector
        vector should be the vector (ndarray)
        n should be the number of values you want displayed in the chart
        style should be the format of the chart
        ascending=False will be most relevant traits by magnitude,
        ascending=True will be least relevant traits by magnitude

        Example: vectordf, plotguy = vector_barchart(trait_columns_relabeled,V[2,:],10)"""
    n = min(n, len(vector_names))
    vectordf = pd.DataFrame()
    vectordf["Trait"] = vector_names
    vectordf["Values"] = vector
    if style == "by_mag":
        vectordf["Magnitude"] = vectordf.apply(lambda row: abs(row["Values"]),
                                               axis=1)
        sorteddf = vectordf.sort_values(by="Magnitude", ascending=ascending)
        # plotguy = sorteddf.iloc[-2*n:].iloc[::-1]
        plotguy = sorteddf.iloc[0:2 * n]
    # sns.set(font_scale = 2)
    sns.barplot(plotguy["Values"], plotguy["Trait"], ci=None)
    # sns.set(font_scale = 1)
    plt.show()
    return vectordf, plotguy


if __name__ == '__main__':
    V = np.load("V.npy")
    data_df = pd.read_json(
        "/Users/jzimmer1/Documents/GitHub/character-space/lsa.json")
    vector_barchart(data_df.columns, V[0, :], 15)
