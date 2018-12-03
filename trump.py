import numpy as np
import csv

X = []
xTr = []
yTr = []
xTe = []
yTe = []

with open('train.csv', 'r') as train:
	reader = csv.reader(train)
	next(reader)
	for row in reader:
		X.append(row[1])
		xTr.append(row[1])
		yTr.append(int(row[17]))


with open('test.csv', 'r') as test:
	reader = csv.reader(test)
	next(reader)
	for row in reader:
		xTe.append(row[1])
		X.append(row[1])

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words='english')
xTr_count = count_vect.fit_transform(xTr)
xTe_count = count_vect.fit_transform(xTe)
X_count = count_vect.fit_transform(X)

from sklearn.feature_extraction.text import TfidfTransformer
Tfid_trans = TfidfTransformer()
xTr_tfid = Tfid_trans.fit_transform(xTr_count)
xTe_tfid = Tfid_trans.fit_transform(xTe_count)
X_tfid = Tfid_trans.fit_transform(X_count)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_tfid[:len(yTr)], yTr)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
prediction = clf.predict(X_tfid[len(yTr):])

with open('submission.csv', mode='w') as submission:
	writer = csv.writer(submission)
	writer.writerow(['ID', 'Label'])
	for i in range(300):
		writer.writerow([str(i), str(prediction[i])])