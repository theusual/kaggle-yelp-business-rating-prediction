__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '07-06-2013'

'''
Data visualizations.
-This module is focused on exploring the data visually.
'''

#------------------
#Visualizations
#------------------
reviewCounts = frmAll[1]['review_count'].value_counts()
reviewCounts[:10].plot(kind='barh',rot=1)
plt.show()

#plot a univariate relationship
X = frmTrn_All.ix[:10000,['calc_rev_age_scaled']].as_matrix()
Y = frmTrn_All.ix[:10000,['rev_votes_useful']].as_matrix()

clf = SGDRegressor(alpha=0.01, n_iter=100,shuffle=True)
clf.fit(X,Y)

pl.scatter(X,Y)
pl.plot(X, clf.predict(X), color='blue', linewidth=3)
pl.show()

   ##Another:
X = frmTrn_All.ix[:10000,['calc_total_checkins_scaled']].as_matrix()
Y = frmTrn_All.ix[:10000,['calc_daily_avg_useful_votes']].as_matrix()

clf = SGDRegressor(alpha=0.01, n_iter=100,shuffle=True)
clf.fit(X,Y)

pl.scatter(X,Y)
pl.plot(X, clf.predict(X), color='blue', linewidth=3)
pl.show()