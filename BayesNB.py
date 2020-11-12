import pandas
import numpy as np
import math
import string

def getTrainData():
	'''General purpose method to generate a dataset from a csv and gathering a list of 
	possible labels for our texts'''
	trainData=pandas.read_csv("train.csv")
	data=[trainData["Abstract"], trainData["Category"]]
	categories={}
	for i in data[1]:
		if i not in categories:
			#encodes the label by an integer
			categories[i]=len(categories)

	return data[0], data[1], categories


def getTestData():
	''' General purpose method to simply load the test data from a csv 
	without any labels'''
	trainData=pandas.read_csv("test.csv")
	data=[trainData["Abstract"]]

	return data[0]



def split_dataset(dataX, dataY, seed):
	''' Splits our data based a specified seed in order to have a specific
	ratio of testset and dataset sizes.'''
	trainX=[]
	trainY=[]
	testX=[]
	testY=[]
	for i in range(len(dataX)):

		if i%5==(seed%5):
			testX.append(dataX[i])
			testY.append(dataY[i])
		else:
			trainX.append(dataX[i])
			trainY.append(dataY[i])

	return [trainX, trainY], [testX, testY]


def treatment(dataX):
	''' Cleaning method in order to filer noise and unregularities'''
	
	for i in range(len(dataX)):
		text = dataX[i].split(" ")#splits each text by words delimited by spaces
		text1=[]
		for j in text:
			#Here we check if there are uncaught spaces and resplit
			if "\n" in j:
				temp = j.split("\n")
				for g in temp:
					text1.append(g)
			else:
				text1.append(j)

		text2=[]
		for j in text1:
			if "-" in j:
				#Here we check if the word is a hyphenated word and split it
				temp = j.split("-")
				for g in temp:
					text2.append(g)
			else:
				text2.append(j)

		text3=[]
		for j in text2:
			#list of non letter caracters
			carToClean=string.punctuation
			for car in carToClean:
				#remove the unwanted caracter
				j=j.replace(car, "")
			text3.append(j)


		text4=[]
		for j in text3:
			#only keeps words that are longer than 3 letters.
			if len(j)>3:
				text4.append(j)

		dataX[i]=text4

	return dataX

class NBbayes():

	def __init__(self):
		
		self.categories=None
		self.conditionals=None
		self.priors=None


	def train(self, trainX, trainY, categories):
		''' method to train our model with a given data set'''

		self.categories=categories
		#gets the number of classes
		count=self.getClassCount(trainY)
		
		Voc={}# vocabulary of words
		for i in range(len(trainX)):
			readWords = {}#words encountered so far
			for word in trainX[i]:
				if word not in readWords.keys():
					readWords[word]=0
					if word not in Voc.keys():
						#given a new word we generate a list of 15 entries
						#This list will contain the number of occurances of a word 
						#for each category encoded by an index.
						Voc[word] = [0]*15
						Voc[word][categories[trainY[i]]] = 1
					else :
						Voc[word][categories[trainY[i]]] += 1

		for key in Voc.keys():
			for cat in self.categories.keys():
				#Method to ensure that there are no 0 values of probability
				# This also transform a count into a probability by dividing by the total
				Voc[key][self.categories[cat]]=(Voc[key][self.categories[cat]]+1)/(count[cat]+len(self.categories))

		self.conditionals=Voc

		for cat in count.keys():
			count[cat]=count[cat]/len(trainX)

		self.priors=count

	def getClassCount(self, trainY): #generates the vector containing the number of abstracts in each category

		count={}
		for cat in self.categories.keys():
			count[cat]=0

		for category in trainY:
			count[category]+=1
		return count


	def makePredictions(self, testX): #this methods classifies the test set using the conditional probabilities
		
		predictions=[]#the list containing the predicted categories.
		
		for doc in testX:#we loop though every abstracts of the test set
			scoreDoc=[0]*len(self.categories) #our predicted probabilities vector

			for word in doc:
				for cat in self.categories.keys():
					if word in self.conditionals.keys():#if the word is not in the features it is not counted
						pX=self.conditionals[word][self.categories[cat]]#the probabilty associated to a specific word
						scoreDoc[self.categories[cat]]+=math.log(pX)#we compute the logarithm of the probability, product
						#would have caused rounding errors. 

			for key in self.priors.keys():#the priors are added
				scoreDoc[self.categories[key]]+=math.log(self.priors[key])

			pred=None

			ind=scoreDoc.index(max(scoreDoc))#the predicted category is the index of the max probability.
			for cat in self.categories.keys():#we loop through the categories to find the matching index
				if ind == self.categories[cat]:
					pred=cat
			predictions.append(pred)

		return predictions

	def computeAccuracy(self, testY, predictions):#this methods returns the success rate of a prediction
		good=0
		for i, label in enumerate(testY):
			if label==predictions[i]:
				good+=1
		return good/len(predictions)


def localTest():#this methods is used to test our model.
	bruteDataX, dataY, categories=getTrainData()
	dataX=treatment(bruteDataX)

	trainSet, testSet=split_dataset(dataX, dataY, 17)#the dataset is split into training and test sets

	bayes=NBbayes()
	bayes.train(trainSet[0], trainSet[1], categories)
	preds=bayes.makePredictions(testSet[0])
	print(bayes.computeAccuracy(testSet[1], preds))

def localAverage():#this method computes the average success rate over all 5 possible seeds

	bruteDataX, dataY, categories=getTrainData()
	dataX=treatment(bruteDataX)
	temp=0
	for i in range(0, 5):
		trainSet, testSet=split_dataset(dataX, dataY, i)

		bayes=NBbayes()
		bayes.train(trainSet[0], trainSet[1], categories)
		preds=bayes.makePredictions(testSet[0])
		temp+=bayes.computeAccuracy(testSet[1], preds)
	print(temp/5)



def predictionsToCSV(preds):#this methods creates the submission file.
	ids=[]
	category=[]
	for i in range(len(preds)):
		ids.append(i)
		category.append(preds[i])
	dik={}
	dik["Category"]=preds
	frame=pandas.DataFrame.from_dict(dik)
	frame.to_csv("predictions", header=True)


def genPredictions():#this methods generates the test predictions.
	bruteTrainX, trainY, categories=getTrainData()
	trainX=treatment(bruteTrainX)

	bruteTestX=getTestData()
	testX=treatment(bruteTestX)#we test on the unlabelled dataset

	bayes=NBbayes()
	bayes.train(trainX, trainY, categories)
	preds=bayes.makePredictions(testX)
	predictionsToCSV(preds)


genPredictions()

#localAverage()
localTest()
