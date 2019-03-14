import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import csv
import time
from numpy import linalg as la
import math
from svmutil import *
import sys
import random
# X = []
# Y = []
# X_test = []
# Y_test = []

train = sys.argv[1]
test = sys.argv[2]
# part = sys.argv[3]

# time_gap = float(sys.argv[4])



X = []
Y = []
X_test = []
Y_test = []

# num = 5

start1 = time.time()

# train = "mnist/train.csv"
# test = "mnist/test.csv"

with open(train) as fileX:
	x_reader = csv.reader(fileX)
	for row in x_reader:
		temp = []
		for i in range(784):
			temp.append(float(row[i])/255)
		Y.append(float(row[784]))
		X.append(temp)


with open(test) as fileX:
	x_reader = csv.reader(fileX)
	for row in x_reader:
		temp = []
		for i in range(784):
			temp.append(float(row[i])/255)
		Y_test.append(float(row[784]))
		X_test.append(temp)

end1 = time.time()
print("Input done, Time taken(sec): ", int(end1-start1))

start2 = time.time()
prob  = svm_problem(Y, X)
param = svm_parameter('-t 2 -c 1 -b 0 -g 0.05 -q')
m = svm_train(prob, param, '-q')
end2 = time.time()
print("Training, Time taken using LIBSVM: ", int(end2-start2))
p_label, p_acc, p_val = svm_predict(Y_test, X_test, m, '-b 0 -q')
p_label1, p_acc1, p_val1 = svm_predict(Y, X, m, '-b 0 -q')
ACC, MSE, SCC = evaluations(Y_test, p_label)
ACC1, MSE1, SCC1 = evaluations(Y, p_label1)
print("Accuracy using LIBSVM on multiclass(on test data): ", ACC)
print("Accuracy using LIBSVM on multiclass(on training data): ", ACC1)

# end2 = time.time()
# print("Time taken(sec)", int(end2-start2))
