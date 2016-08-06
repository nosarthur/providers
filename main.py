import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN, MiniBatchKMeans, SpectralClustering
from sklearn.cluster import AgglomerativeClustering, MeanShift
from sklearn.cluster import AffinityPropagation, Birch
from sklearn.cluster import estimate_bandwidth
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier 
import utils    # user defined functions

''' ------------ load data ---------------- '''
exams = pd.read_csv('ExamRatingData072516.csv')
specs = pd.read_csv('ProviderSubspecializationData072516.csv')
scanners = pd.read_csv('ProviderEquipmentData072516.csv')

''' ------------ data cleaning ------------ '''
# impute RadPeer.Significance.of.Errors from RadPeer.w.Significance.Score
filter0 = exams['RadPeer.Significance.of.Errors'].isnull()

# treat score 1 
filter1 = (exams['RadPeer.w.Significance.Score']=='1') & filter0 
exams.loc[filter1, 'RadPeer.Significance.of.Errors'] = 0

# treat two letter scores 
tmp = exams[exams['RadPeer.Significance.of.Errors'].isnull()]
score = tmp['RadPeer.w.Significance.Score'].apply(utils.extract_ab)
exams.loc[(score=='a').index, 'RadPeer.Significance.of.Errors'] = 0
exams.loc[(score=='b').index, 'RadPeer.Significance.of.Errors'] = 1

# average reviews for the same exam
columns = ['Exam.Quality.Reviewer.ID', 'RadPeer.w.Significance.Score',
           'Patient.Sex', 'Patient.Age', 'study_body_part', 
           'MSK.or.Spine']
exams.drop(columns, axis=1, inplace=True)

exams = exams.groupby(['Exam.ID', 'Provider.ID']).agg('mean')
exams['Percent.Error'] = exams['Total.Diagnostic.Errors'] \
              / (exams['Total.Diagnostic.Errors']
               + exams['True.Positive.Count'] 
               + exams['True.Negative.Count'])

''' ---- Task 1: quality review summary for each provider ---- '''
summary = exams.groupby(level='Provider.ID').agg('mean')

# impute 'Percent.Error' column
utils.impute_error(summary)

# save summary to file 
summary.to_csv('summary.csv')

''' ---- Task 2: clustering according to "care quality" ---- '''
columns = ['RadPeer.Score', 'RadPeer.Significance.of.Errors',
 			'Technical.Performance.Score', 'Percent.Error']
features = summary[columns]
#fig = pd.scatter_matrix(features, figsize=(18,18), alpha=0.5, grid=True)
#sns.plt.savefig('features_scatter.png', bbox_inches='tight')

# scaling
mms = MinMaxScaler()
X = mms.fit_transform(features)

db = DBSCAN(eps=0.3, min_samples=5)
ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean',
                             linkage='average')
km = MiniBatchKMeans(n_clusters=2, random_state=1, n_init=15)
bc = Birch(n_clusters=2)
#sp = SpectralClustering(n_clusters=2, eigen_solver='arpack', random_state=1) 
#bandwidth = estimate_bandwidth(X, quantile=0.3)
#ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
#ap= AffinityPropagation(damping=.9, preference=-200)

y_km = km.fit_predict(X)
y_ac = ac.fit_predict(X)
utils.swap_label(y_ac)
#y_bc = bc.fit_predict(X)
#utils.swap_label(y_bc)
y_db = db.fit_predict(X)
y_db[y_db==-1] = 1
#print np.unique(y_db)
#y_sp = sp.fit_predict(X)
#y_ms = ms.fit_predict(X)
#y_ap = ap.fit_predict(X)

labels = {'AgglomerativeClustering':y_ac}
labels['MiniBatchKMeans'] = y_km 
labels['DBSCAN'] = y_db
#labels['Birch'] = y_bc

fig, axes = plt.subplots(3,len(labels), figsize=(17,10))
for key, value in labels.items():
    print("Silhouette Coefficient of %s: %0.3f"
              % (key, silhouette_score(X, value)) )
cols = ['RadPeer.Score', 'Technical.Performance.Score']
utils.plot_labelled_columns(features, cols, labels, axes[0], with_title=True)

cols = ['RadPeer.Score', 'Percent.Error']
utils.plot_labelled_columns(features, cols, labels, axes[1])

cols = ['RadPeer.Score', 'RadPeer.Significance.of.Errors']
utils.plot_labelled_columns(features, cols, labels, axes[2], with_xlabel=True)
#sns.plt.show()
sns.plt.savefig('clustering_compare.png', bbox_inches='tight')

# make dendrogram and heat plots
#print(features.columns.tolist())
#utils.plot_dendr_heat(features, labels)

# add labels column
features = features.assign(label=y_ac)

''' ------- Task 3: prediction of "care quality" ------- '''

# add features from scanner specs
gp = scanners.groupby('Provider.ID')
new_f = gp['MRI.magnet.strength'].agg([('B0.max','max'), 
                                ('B0.min','min'), 
                                ('scanner.count','size')])
d = {'Closed':'C', 'Semi-Open Wide Bore':'W', 'Open':'O',
     'Stand-Up':'S', 'Extremity':'E'}  # shorten the name string
# add up the name strings if the provider has more than 1 scanner
mType = gp[['MRI.machine.type']].agg(lambda s: s.apply(lambda x: d[x]).sum())

# inner join, 1 provider drops out
df = pd.merge(new_f, specs, right_on='Provider.ID', left_index=True) 
df = pd.merge(mType, df, right_on='Provider.ID', left_index=True) 
df = pd.merge(features[['label']], df, right_on='Provider.ID',
              left_index=True)
df.set_index('Provider.ID', inplace=True)

# plot provider 'goodness' WRT individual features 
#utils.explore_features(df)

# turn categorical columns to one-hot-encoding
df = pd.get_dummies(df)

# 3-fold cross validation 
predictors = df.columns.tolist()
predictors.remove('label')

clf = RandomForestClassifier(class_weight='balanced', random_state=1, bootstrap=True,oob_score=True,
                             max_features='auto', n_estimators=10000,
                             min_samples_split=2, min_samples_leaf=2)
scores = cross_val_score(clf, df[predictors], df['label'], cv=3, 
                         scoring='f1')
print('CV accuracy: %.3f +/- %.3f' % (scores.mean(), scores.std()))

# fit full dataset to get ROC
clf.fit(df[predictors], df['label'])

y_pred = clf.predict_proba(df[predictors])[:,1]
utils.plot_ROC(df['label'], y_pred)

# feature importance
utils.plot_feature_importances(clf.feature_importances_)

# 
yy = clf.predict(df[predictors])
df = df.assign(predicted_label=yy)
check = df[['predicted_label']]
lala = pd.merge(check, features, left_index=True, right_index=True)

    

