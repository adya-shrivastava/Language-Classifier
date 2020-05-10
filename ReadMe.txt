Intro to IS Lab 2: Wikipedia Language Classification

A classifier to classify sentence as either Dutch ('nl') or English ('en').

I have used two learning algorithms for this purpose:
1. Decision Tree Algorithm
2. Adaboost Algorithm

How to run the program:
1. Training the model:
	To train the model, use 'train.py' with training set, hypothesis_file and the learning algorithm to be used. 
	For example:     python3 train <examples> <hypothesisOut> <learning-type> where learning-type is either 'dt' for decision tree of 'ada' for Adaboost

2. Testing the model:
	To test the model on a testing file, use 'predict.py' with hypothesis_file and test_file 
	For example: 	python3 predict <hypothesis> <file> where hypothesis is the file containing the trained model
	
	Output: n lines (either 'en' or 'nl') which prints the predicted class label.