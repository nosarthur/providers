from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import roc_curve, auc

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
        sns.plt.savefig('impute_error.png', bbox_inches='tight')

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

def explore_features(df):
    by_state = df.groupby(['provider_state', 'label']).size().unstack()
    ax = by_state.plot(kind='barh', stacked=True)
    ax.set(xlabel='Provider count', ylabel='Provider State')
    sns.plt.savefig('by_state.png', bbox_inches='tight')

    by_count = df.groupby(['scanner.count', 'label']).size().unstack()
    ax = by_count.plot(kind='barh', stacked=True)
    ax.set(xlabel='Provider count', ylabel='Provider scanner count')
    sns.plt.savefig('by_scanner_count.png', bbox_inches='tight')

    by_max_B0 = df.groupby(['B0.max', 'label']).size().unstack()
    ax = by_max_B0.plot(kind='barh', stacked=True)
    ax.set(xlabel='Provider count', ylabel='Max Scanner Strength')
    sns.plt.savefig('by_max_B0.png', bbox_inches='tight')

#    by_avg_B0 = df.groupby(['B0.avg', 'label']).size().unstack()
#    ax = by_avg_B0.plot(kind='barh', stacked=True)
#    ax.set(xlabel='Provider count', ylabel='Avg Scanner Strength')
#    sns.plt.savefig('by_avg_B0.png', bbox_inches='tight')

    by_min_B0 = df.groupby(['B0.min', 'label']).size().unstack()
    ax = by_min_B0.plot(kind='barh', stacked=True)
    ax.set(xlabel='Provider count', ylabel='Min Scanner Strength')
    sns.plt.savefig('by_min_B0.png', bbox_inches='tight')

    by_subspec= df.groupby(['Is.Subspecialized', 'label']).size().unstack()
    ax = by_subspec.plot(kind='barh', stacked=True)
    ax.set(xlabel='Provider count', ylabel='Is.Subspecialized')
    sns.plt.savefig('by_subspec.png', bbox_inches='tight')

    by_mType = df.groupby(['MRI.machine.type', 'label']).size().unstack()
    ax = by_mType.plot(kind='barh', stacked=True)
    ax.set(xlabel='Provider count', ylabel='MRI.machine.type')
    sns.plt.savefig('by_mType.png', bbox_inches='tight')

def plot_by_type(df, scanners):
    tmp = pd.merge(scanners[['Provider.ID', ]], df[['label']], left_on='Provider.ID', right_index=True)
    tmp.set_index('Provider.ID', inplace=True)

    by_type = tmp.groupby(['MRI.machine.type','label']).size().unstack()
    ax = by_type.plot(kind='barh', stacked=True)
    ax.set(xlabel='Provider count', ylabel='Machine type')
    sns.plt.savefig('by_type.png', bbox_inches='tight')

def plot_ROC(y_true, y_score):
    fpr, tpr, thresh = roc_curve(y_true=y_true, y_score=y_score)
    roc_auc = auc(x=fpr, y=tpr)
    fig2 = plt.figure()
    plt.plot(fpr, tpr, label='auc = %0.2f'%roc_auc) 
    plt.plot([0,1],[0,1],linestyle='--', c='gray', linewidth=2)
    plt.legend(loc='best')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('Receiver Operator Characteristic')
    sns.plt.savefig('ROC.png', bbox_inches='tight')
#plt.show()


def plot_feature_importances(importances):

    indices = np.argsort(importances)[::-1] # descending
    p_names = ['B0.max', 'B0.min', 'scanner.count', 
               'specialized', 'type.C', 'type.CC',
               'type.CE', 'type.CS', 'type.O', 'type.OC', 'type.OCC', 
               'type.OO', 'type.OW', 'type.S', 'type.SE', 'type.W',
               'type.WC', 'type.WE', 'type.WW', 'state.AL', 'state.CA',
               'state.CT', 'state.FL', 'state.GA', 'state.KS', 'state.MD',
               'state.MO', 'state.NC', 'state.NJ', 'state.NY', 'state.PA',
               'state.SC', 'state.TN', 'state.VA']
    n_features = len(p_names)
    for i in range(n_features):
        print('%2d) %-*s %f' % (i+1, 30, p_names[i], 
                                importances[indices[i]]))
    # make plot
    fig = plt.figure()
    plt.bar(range(n_features), importances[indices], 
            color='blue', align='center', alpha=0.6)
    plt.xticks(range(n_features), p_names, rotation=90)
    plt.xlim([-1, n_features])
    plt.title('Feature Importances')
    #plt.show()
    sns.plt.savefig('feature_importance.png', bbox_inches='tight')


