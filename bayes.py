import utils as ut
import numpy as np
import sys
import string
import math
import random

TESTSIZE = 133718
TRAINSIZE = 534872


# def remove_duplicates(x):
# 	return list(dict.fromkeys(x))

def main():	
	#Making Vocabulary for different labels out of training data
	vocab_list = [{}, {}, {}, {}, {}]
	vocabulary = {}
	#Count of each label in training data
	label_count = np.zeros(5)
	label_word_count = np.zeros(5)
##############################################################################
	#Training part
	iter = (ut.json_reader("train.json"))
	for i in range(TRAINSIZE):
		if (i%1000)==0:
			print("Training: ", i/1000)
		# for i in range(1):
		element = next(iter)
		label_count[int(element["stars"])-1]+=1
		# print((remove_duplicates((element["text"]).split())))
		label_word_count[int(element["stars"])-1]+= len((element["text"]).split())
		# Switch these lines for stemming 
		# for x in ((element["text"]).split()):
		# stemmed = ut.getStemmedDocuments(element["text"])
		for x in (ut.getStemmedDocuments(element["text"])):
			word = x.strip(string.punctuation)
			# word = x
			# print(word)
			if word=="":
				continue
			if word in vocab_list[int(element["stars"]-1)]: 
				(vocab_list[int(element["stars"])-1])[word]+=1
			else:
				(vocab_list[int(element["stars"])-1])[word]=1
	
			vocabulary[word]=1


	# for x in vocabulary:
	# 	for i in range(5):
	# 		if x in vocab_list[i]:



##############################################################################



	# print(len(vocab))
	# count=0;
	# for i in range(5):
	# 	print(label_count[i])
	# 	count+=(label_count[i])
	# print(count)
	prior = label_count/TRAINSIZE
	# print(prior)
	
	actual_value = []
	predicted_value = []
	random_prediction = []
##############################################################################
	#TESTING
	iter2 = (ut.json_reader("test.json"))
	for i in range(TESTSIZE):
		if (i%1000)==0:
			print("Testing: ", i/1000)
		# print(i)
		#Random number between 1-5
		random_prediction.append(random.randint(1,6))
		test_element = next(iter2)
		actual_value.append(int(test_element["stars"]))
		# test = "Stopped here today to give it a try and must admit the food was excellent. I ordered the vegetarian Soyrizo (fake sausage) burrito and fell in love. It was well worth the $6. It's not like the big chain restaurants where they serve you a massive sloppy burrito. It was the perfect size and easily handled. \nIt's small and quaint, with some seating outside in under a canopy. The owners were a lovely couple, passionate about their food. \nExcellent."
		# test = "Fast, easy, helpful. In and out quickly and got the medicine I needed. Smart staff who was kind and helpful. Clean facility. No complaints from me"	
		# test = "Service good, we had hummas, gyros, spiced date crumble.... all real good... need to try the flamming cheese next time!...  messed up on a few tables bill.. including ours but got it fixed.  I liked it. . .  my guest was on the fence."
		test = test_element["text"]
		# test_list = ((test).split())
		test_list = (ut.getStemmedDocuments(test_element["text"]))
		# print(test_list)
		results = []
		for i in range(5):
		#check for 1 rating
		# i=0
			py = prior[i]
			logr = 0
			rating=i+1
			for x in test_list:
				word = x.strip(string.punctuation)
				# word = x
				# print(word)
				if word == "":
					continue
				if word in vocab_list[i]:
					# print(word)
					# print(((vocab_list[i])[word]))
					# print(label_count[i])
					probability = (((vocab_list[i])[word])+1)/(label_word_count[i]+len(vocabulary))
					logr+=math.log(probability)
				else:
					# print("not")
					logr+=math.log(1/(label_word_count[i]+len(vocabulary)))
			results.append(logr+(math.log(py)))
			# print("------------------------------------------")
		
		predicted_value.append(results.index(max(results))+1)
		# print(results.index(max(results))+1)
##############################################################################

	# print(len(predicted_value))

	major = list(label_count).index(max(label_count))+1
	correct=0
	correct_random=0
	correct_major=0;
	confusion =  np.zeros((5,5))

	for i in range(len(predicted_value)):
		# print(predicted_value[i])
		if(predicted_value[i]==actual_value[i]):
			correct+=1
		if(random_prediction[i]==actual_value[i]):
			correct_random+=1
		if(major==actual_value[i]):
			correct_major+=1
		confusion[predicted_value[i]-1][actual_value[i]-1]+=1
	
	# print("Correct")
	# print(correct)
	# print(len(actual_value))
	print("Accuracy using Naive Bayes: ", int(correct/len(actual_value)*100) , "%")
	print("Accuracy using Random prediciton: ", int(correct_random/len(actual_value)*100) , "%")
	print("Accuracy using Majority prediciton: ", int(correct_major/len(actual_value)*100) , "%")
	print("Confusion Matrix: ")
	print(confusion)
	

	# new_text = "It is important to by very pythonly while you are pythoning with python.All pythoners have pythoned poorly at least once."
	# print(new_text)
	# print(ut.getStemmedDocuments(new_text))





if __name__ == '__main__':
    main()