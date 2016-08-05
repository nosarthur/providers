from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage

def extract_ab(row):
    return row[1]

def impute_error(df, is_plot=False):
    slr = LinearRegression()
    mask1 = df['Percent.Error'].notnull()
    mask0 = df['Percent.Error'].isnull()
    df1 = df[mask1]
    df0 = df[mask0]
    print df0.shape, df1.shape

    # linear regression 
    slr.fit(df1[['RadPeer.Score']], df1['Percent.Error'])
    predicted1 = slr.predict(df.loc[mask1, ['RadPeer.Score']])
    predicted0 = slr.predict(df.loc[mask0, ['RadPeer.Score']])

    df.loc[mask0, 'Percent.Error'] = predicted0

    if is_plot: # make plot 
        df1.plot(kind='scatter', x='RadPeer.Score', y='Percent.Error',
                 color='blue', alpha=0.4, label='126 non-null',
                 figsize=(7,7), zorder=2)
        plt.plot(df1[['RadPeer.Score']], predicted1, color='blue',
                 label='linear fit', zorder=1) 	
        plt.scatter(df0[['RadPeer.Score']], predicted0, color='red',
                    alpha=0.6, label='71 null', zorder=3) 	

        plt.legend(loc='upper left')
        #sns.plt.show()
        sns.plt.savefig('impute.png', bbox_inches='tight', pad_inches=0)

def plot_labelled_columns(df, cols, labels, axes, ms='os', 
            cs=['orange', 'blue'], with_title=False, with_xlabel=False):
    xs, ys = [[],[]], [[],[]]
    for label in (0,1):
        for y in labels.values():
            xs[label].append(df.loc[y==label, cols[0]])
            ys[label].append(df.loc[y==label, cols[1]])
    for i in range(len(labels)):
        axes[i].scatter(xs[0][i], ys[0][i], alpha=0.4, c=cs[0], marker=ms[0])
        axes[i].scatter(xs[1][i], ys[1][i], alpha=0.4, c=cs[1], marker=ms[1])
        if with_xlabel:
            axes[i].set_xlabel(cols[0])
        if with_title:
            axes[i].set_title(labels.keys()[i])
    axes[0].set_ylabel(cols[1])

def swap_label(y):
    y[y==0] = 2
    y[y==1] = 0
    y[y==2] = 1

def plot_dendr_heat(df, labels):
    row_clusters = linkage(df.values, method='average', metric='euclidean')
    row_dendr = dendrogram(row_clusters, labels=labels['AgglomerativeClustering'])
    plt.tight_layout()
    plt.ylabel('Euclidean distance')
    plt.show()

