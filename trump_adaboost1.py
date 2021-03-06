import csv

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
		xTr.append(row[1])
		yTr.append(int(row[17])) # Column 17 is the label cell

# Read in the test data
with open('test.csv', 'r') as test:
	reader = csv.reader(test)
	next(reader)
	for row in reader:
		xTe.append(row[1])
		X.append(row[1])

# Transform the vector of tweets to matrix representing
# all tweets in the BOW(bag of words) form
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(stop_words = "english")
X = tfidf_vect.fit_transform(X)
xTr = X[:len(yTr)]
xTe = X[len(yTr):]

# Using adaboost to classify
from sklearn.ensemble import AdaBoostClassifier
classifier = AdaBoostClassifier(n_estimators=60).fit(xTr, yTr)
yTe = classifier.predict(xTe)

with open('submission_adaboost1.csv', mode='w') as submission:
	writer = csv.writer(submission)
	writer.writerow(['ID', 'Label'])
	for i in range(300):
		writer.writerow([str(i), str(yTe[i])])
