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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
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
#sns.plt.savefig('features_scatter.png', bbox_inches='tight', pad_inches=0)

# scaling
mms = MinMaxScaler()
X = mms.fit_transform(features)

db = DBSCAN(eps=0.3, min_samples=5)
ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean',
                             linkage='average')
km = MiniBatchKMeans(n_clusters=2, random_state=42, n_init=15)
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

labels = {'MiniBatchKMeans':y_km, 'AgglomerativeClustering':y_ac}
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
sns.plt.savefig('clustering_compare.png', bbox_inches='tight', pad_inches=0)

# make dendrogram and heat plots
print(features.columns.tolist())
#utils.plot_dendr_heat(features, labels)

# add labels column
features = features.assign(label=y_ac)

''' ------- Task 3: prediction of "care quality" ------- '''

# inner join, 1 provider drops out
df = pd.merge(scanners, specs[['Provider.ID','Is.Subspecialized']], on='Provider.ID') 
#df.set_index('Provider.ID', inplace=True)

# add features
df = pd.merge(features[['label']], df, right_on='Provider.ID',
              left_index=True)

df = pd.get_dummies(df)

# cross validation 
clf1 = LogisticRegression(C=1, random_state=42, penalty='l2', class_weight='balanced')
clf2 = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced',
                min_samples_split=1, min_samples_leaf=2, max_features=4)

predictors = df.columns.tolist()
predictors.remove('label')

scores = cross_val_score(clf1, df[predictors], df['label'], cv=5, scoring='f1')
print 'logistic:', scores.mean()
scores = cross_val_score(clf2, df[predictors], df['label'], cv=5, scoring='f1')
print 'random forest:', scores.mean()

eclf = VotingClassifier([('lr', clf1), ('rf', clf2)], voting='soft')
scores = cross_val_score(eclf, df[predictors], df['label'], cv=5, scoring='f1')


print('CV accuracy: %.3f +/- %.3f' % (scores.mean(), scores.std()))

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

eclf.fit(df[predictors], df['label'])
#y_pred = eclf.predict_proba(df[predictors])[:,1]
y_pred = clf1.fit(df[predictors], df['label']).predict_proba(df[predictors])[:,1]

fpr, tpr, thresh = roc_curve(y_true=df['label'], y_score=y_pred)
roc_auc = auc(x=fpr, y=tpr)
fig2 = plt.figure()
plt.plot(fpr, tpr, label='auc = %0.2f'%roc_auc) 
plt.plot([0,1],[0,1],linestyle='--', c='gray', linewidth=2)
plt.legend(loc='best')
plt.show()

    

