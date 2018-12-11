import csv

x = []
y = []

with open('test.csv', 'r') as test:
	reader = csv.reader(test)
	next(reader)
	for row in reader:
		x.append(row[1])

r = []

with open('1.csv', 'r') as results:
	reader1 = csv.reader(results)
	next(reader1)
	for row in reader1:
		r.append(row[1])

for i in range(300):
    indicator = 1
    for j in range(2346):
        if (x[i] == r[j]):
            indicator = -1
            break
    y.append(indicator)

with open('312.csv', mode='w') as submission:
	writer = csv.writer(submission)
	writer.writerow(['ID', 'Label'])
	for i in range(300):
		writer.writerow([str(i), str(y[i])])
