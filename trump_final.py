import csv

import numpy as np
from textblob import TextBlob
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

X = []
X_char = []
X_word = []
X_sentiment = []
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
		sentiment = TextBlob(row[1]).sentiment
		X_sentiment.append([sentiment.polarity, sentiment.subjectivity])

# Read in the test data
with open('test.csv', 'r') as test:
	reader = csv.reader(test)
	next(reader)
	for row in reader:
		X.append(row[1])
		sentiment = TextBlob(row[1]).sentiment
		X_sentiment.append([sentiment.polarity, sentiment.subjectivity])

# Sentiment analysis


# Transform the vector of tweets to matrix representing
# All tweets in the BOW(bag of words) form
tfidf_vect_char = TfidfVectorizer(stop_words = "english", min_df=3, max_df=0.8, sublinear_tf=True, use_idf=True, analyzer='char')
X_char = tfidf_vect_char.fit_transform(X)
print(X_char.shape)
tfidf_vect_word = TfidfVectorizer(stop_words = "english", min_df=3, max_df=0.8, sublinear_tf=True, use_idf=True)
X_word = tfidf_vect_word.fit_transform(X)
print(X_word.shape)
X = np.concatenate((np.array(X_char.toarray()), np.array(X_word.toarray())), axis=1)
print(np.array(X).shape)

#X = np.concatenate((np.array(X.toarray()), np.array(X_sentiment)), axis=1)
xTr = X[:len(yTr)]
xTe = X[len(yTr):]

# Use Gridsearch to find optimal hyperparameter
clf0 = SVC(probability=True, C=0.1, kernel='rbf', degree=3, gamma='auto')
clf1 = SVC(probability=True,C=10, kernel='rbf', degree=3, gamma='auto')
clf2 = SVC(probability=True,C=1.0, kernel='rbf', degree=3, gamma='auto')
clf3 = SVC(probability=True,C=0.01, kernel='rbf', degree=3, gamma='auto')
clf4 = SVC(probability=True,C=0.001, kernel='rbf', degree=3, gamma='auto')
clf5 = DecisionTreeClassifier(criterion='gini', max_depth=1)
clf6 = LogisticRegression(solver='lbfgs', C = 0.1)
clf7 = DecisionTreeClassifier(criterion='gini', max_depth=2)
clf8 = DecisionTreeClassifier(criterion='gini', max_depth=27)
clf9 = DecisionTreeClassifier(criterion='gini', max_depth=25)
clf10 = DecisionTreeClassifier(criterion='gini', max_depth=30)
clf11 = DecisionTreeClassifier(criterion='gini', max_depth=35)


param_grid = {
	"base_estimator": [clf5, clf7, clf8, clf9, clf10, clf11],
    "n_estimators": [30, 45, 50, 60, 65],
	"learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2]
}


# Train classifier and make prediction
classifier = AdaBoostClassifier()
grid_search = GridSearchCV(classifier, param_grid=param_grid, scoring='accuracy', refit=True)
grid_search.fit(xTr, yTr)
print(grid_search.best_params_)
yTe = grid_search.predict(xTe)

# Write submission file
with open('submission_charwordbfff.csv', mode='w') as submission:
	writer = csv.writer(submission)
	writer.writerow(['ID', 'Label'])
	for i in range(300):
		writer.writerow([str(i), str(yTe[i])])
