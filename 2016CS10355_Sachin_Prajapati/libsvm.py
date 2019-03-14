import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import csv
import time
from numpy import linalg as la
import math
from svmutil import *
import sys
# X = []
# Y = []
# X_test = []
# Y_test = []

train = sys.argv[1]
test = sys.argv[2]
part = sys.argv[3]

# time_gap = float(sys.argv[4])



Xq1 = []
Yq1 = []
Xq1_test = []
Yq1_test = []

num = 5

start1 = time.time()

# train = "mnist/train.csv"
# test = "mnist/test.csv"

with open(train) as fileX:
	x_reader = csv.reader(fileX)
	for row in x_reader:
		if float(row[784])==num or float(row[784])==num+1: 
			temp = []
			for i in range(784):
				temp.append(float(row[i])/255)
			# Y.append(float(row[784]))
			# X.append(temp)
			if(float(row[784])==num):
				Yq1.append(1)
				Xq1.append(temp)
			if(float(row[784])==num+1):
				Yq1.append(-1)
				Xq1.append(temp)

with open(test) as fileX:
	x_reader = csv.reader(fileX)
	for row in x_reader:
		if float(row[784])==num or float(row[784])==num+1: 
			temp = []
			for i in range(784):
				temp.append(float(row[i])/255)
			# Y.append(float(row[784]))
			# X.append(temp)
			if(float(row[784])==num):
				Yq1_test.append(1)
				Xq1_test.append(temp)
			if(float(row[784])==num+1):
				Yq1_test.append(-1)
				Xq1_test.append(temp)

end1 = time.time()
print("Input done, Time taken", end1-start1)
start2 = time.time()
alpha_count = len(Xq1)


if(part=="c"):
	x_svm, y_svm = Xq1, Yq1

	prob  = svm_problem(y_svm, x_svm)
	param = svm_parameter('-t 2 -c 1 -b 0 -g 0.05 -q')
	# param = svm_parameter('-t 0 -c 1 -b 0 -q')
	m = svm_train(prob, param, '-q')
	p_label, p_acc, p_val = svm_predict(Yq1_test, Xq1_test, m, '-b 0 -q')
	# print("Accuracy using LIBSVM: ", p_acc)
	ACC, MSE, SCC = evaluations(Yq1_test, p_label)
	print("Accuracy using LIBSVM(using guassian kernels): ", ACC)
	alpha_libsvm = m.get_sv_coef()
	SV_indices = m.get_sv_indices()
	print("No. of Support Vectors: ", len(SV_indices))
	alpha_svm = []
	j=0
	for i in range(len(Yq1)):
		if(j<len(SV_indices) and i==SV_indices[j]):
			alpha_svm.append(alpha_libsvm[j][0])
			j+=1
		else:
			alpha_svm.append(0)

	# print(alpha_svm)
	# print(len(alpha_svm))

	matXq1 = np.array(Xq1)
	matYq1 = np.array([Yq1])	
	matAplha = np.array([alpha_svm])

	Wq1 = (matXq1.transpose()).dot((matAplha.transpose())*(matYq1.transpose()))
	bq1 = Yq1[SV_indices[0]] - (Wq1.transpose().dot(matXq1[SV_indices[0]])) 
	# print(Wq1)	
	# print(bq1)


	prob  = svm_problem(y_svm, x_svm)
	param = svm_parameter('-t 0 -c 1 -b 0  -q')
	# param = svm_parameter('-t 0 -c 1 -b 0 -q')
	m = svm_train(prob, param, '-q')
	p_label, p_acc, p_val = svm_predict(Yq1_test, Xq1_test, m, '-b 0 -q')
	# print("Accuracy using LIBSVM: ", p_acc)
	ACC, MSE, SCC = evaluations(Yq1_test, p_label)
	print("Accuracy using LIBSVM(using linear kernels): ", ACC)
	alpha_libsvm = m.get_sv_coef()
	SV_indices = m.get_sv_indices()
	print("No. of Support Vectors: ", len(SV_indices))
