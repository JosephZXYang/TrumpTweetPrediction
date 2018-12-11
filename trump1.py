import csv

from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

X = []
xTr = []
yTr = []
xTe = []
yTe = []

# Read in the training data
with open('train.csv', 'r') as train:
	reader = csv.reader(train)
	next(reader)
	for row in reader:
		X.append(row[1]) # Column 1 is the text cell
		yTr.append(int(row[17])) # Column 17 is the label cell

# Read in the test data
with open('test.csv', 'r') as test:
	reader = csv.reader(test)
	next(reader)
	for row in reader:
		X.append(row[1])

# Transform the vector of tweets to matrix representing
# all tweets in the BOW(bag of words) form
tfidf_vect = TfidfVectorizer(stop_words = "english")
X = tfidf_vect.fit_transform(X)
xTr = X[:len(yTr)]
xTe = X[len(yTr):]

# Use Gridsearch to find optimal hyperparameter

## Gridsearch 1
"""clf1 = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=1)
clf2 = DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=1)
clf3 = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=1)
clf4 = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=1)
clf5 = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None)
clf6 = DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=None)
clf7 = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None)
clf8 = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=None)

param_grid = {
	"base_estimator": [clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8],
    "n_estimators": [5, 10, 20, 100, 50, 75],
	"learning_rate": [1, 2, 3, 4, 5, 6, 7, 0.1, 0.01, 0.05, 0.2, 0.5]
}"""

## Gridsearch 2
"""clf1 = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=1)
clf2 = DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=1)
clf3 = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=1)
clf4 = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=1)
clf5 = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None)
clf6 = DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=None)
clf7 = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None)
clf8 = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=None)"""

"""param_grid = {
	#"base_estimator": [clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8],
    "n_estimators": [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
	"learning_rate": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
}
"""

## Gridsearch 3
clf0 = DecisionTreeClassifier(max_depth=None)
clf1 = DecisionTreeClassifier(max_depth=1)
clf2 = DecisionTreeClassifier(max_depth=2)
clf3 = DecisionTreeClassifier(max_depth=3)
clf4 = DecisionTreeClassifier(max_depth=4)

param_grid = {
	"base_estimator": [clf0, clf1, clf2, clf3, clf4],
    "n_estimators": [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
	"learning_rate": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
}

classifier = AdaBoostClassifier()
grid_search = GridSearchCV(classifier, param_grid=param_grid, scoring='accuracy', refit=True)
grid_search.fit(xTr[:800], yTr[:800])
yTe = grid_search.predict(xTr[800:])
#print(grid_search.best_params_)



# Write submission file
with open('submission_adaboost003.csv', mode='w') as submission:
	writer = csv.writer(submission)
	writer.writerow(['ID', 'Label'])
	for i in range(210+79):
		writer.writerow([str(i+800), str(yTe[i])])
