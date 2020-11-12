import pandas
import numpy as np
import math
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
import string
from nltk.corpus import stopwords 
from heapq import nlargest

class DataSet:
	
	def __init__(self,seed, mode=0):
		self.seed=seed%5
		self.categories=None
		self.catCount=None
		self.Voc=None
		self.dataMatrix=None
		self.trainSet=None
		self.validSet=None
		self.testSet=None
		self.cleaned=None
		if mode==0:
			self.initTest()
		elif mode==1:
			self.initTest2()


	def initTest2(self):

		brute=self.getTrainData()
		categories=brute[1]
		abstractSet=brute[0]
		cleaned=self.format(abstractSet)
		dataSet=cleaned
		self.cleaned=cleaned

		trainSet=self.cleaned

		self.trainSet=trainSet


		Voc=self.genVoc(trainSet, categories)
		priors=self.getCatCount(trainSet, categories)

		self.Voc=Voc
		print(len(self.Voc))
		self.categories=categories

		self.matrix=self.genMatrix(trainSet, Voc, categories)	

		self.catCount=priors	


	def initTest(self):

		brute=self.getTrainData()
		categories=brute[1]
		abstractSet=brute[0]
		cleaned=self.format(abstractSet)
		dataSet=cleaned
		self.cleaned=cleaned

		trainSet, validSet, testSet=self.split_dataset(dataSet)

		self.trainSet=trainSet
		self.validSet=validSet
		self.testSet=testSet

		Voc=self.genVoc(trainSet, categories)
		priors=self.getCatCount(trainSet, categories)

		self.Voc=Voc
		print(len(self.Voc))
		self.categories=categories

		self.matrix=self.genMatrix(trainSet, Voc, categories)

		#self.Voc, self.matrix=self.preTreatment(Voc, self.matrix)

		self.catCount=priors

	def reduceWords(self, n):
		newVoc=self.getNmax(n)
		self.Voc=newVoc
		print(len(self.Voc))
		self.matrix=self.genMatrix(self.trainSet, self.Voc, self.categories)


	def getTrainData(self):
		trainData=pandas.read_csv("train.csv")
		data=[trainData["Abstract"], trainData["Category"]]
		categories={}
		for i in data[1]:
			if i not in categories:
				categories[i]=len(categories)

		return (data,categories)

	def formatQuick(self, data):

		newFormat=[]
		for i in range(len(data[0])):
			
			text = data[0][i].split(" ")
			newEntry=[text, data[1][i]]
			newFormat.append(newEntry)
					
		return newFormat

	def format(self,data):

		lem=WordNetLemmatizer()
		stem=PorterStemmer()

		newFormat=[]
		for i in range(len(data[0])):
			text = data[0][i].split(" ")
			text1=[]
			test="test, test"
			for j in text:
				if "\n" in j:
					temp = j.split("\n")
					for g in temp:
						text1.append(g)
				else:
					text1.append(j)

			text2=[]
			for j in text1:
				if "-" in j:
					temp = j.split("-")
					for g in temp:
						text2.append(g)
				else:
					text2.append(j)

			text3=[]

			for j in text2:
				#carToClean=["$", "(", ")", "{", "}", "\\", "/", "[", "]", ":", ";", ",", "."]
				carToClean=string.punctuation
				for car in carToClean:
					j=j.replace(car, "")

				j=j.lower()
				j=stem.stem(j)

				text3.append(j)

			text4=[]
			for j in text3:
				if j not in stopwords.words('english'):
					text4.append(j)



			newEntry=[text4, data[1][i]]
			newFormat.append(newEntry)
					
		return newFormat

	def split_dataset(self,dataSet):
	    splitTrain1=[]
	    splitValid1=[]
	    splitTest1=[]
	    for i in range(len(dataSet)):

	        if i%5==self.seed%5:
	            splitValid1.append(dataSet[i])
	        elif i%5==(self.seed+1)%5:
	            splitTest1.append(dataSet[i])
	        else:
	        	splitTrain1.append(dataSet[i])

	    return (splitTrain1, splitValid1, splitTest1)

	def genVoc(self,trainingSet,categories):
		Voc={}
		index=0
		for i in range(len(trainingSet)):
			#print(i)
			for word in trainingSet[i][0]:
				if word not in Voc.keys():
					if len(word)>0:
						Voc[word] = index
						index+=1

		return Voc


	def probMatrix(self, matrix):

		conditionalProbs=np.zeros([len(self.categories), len(self.Voc)])

		for pair in matrix:
			conditionalProbs[pair[1]]=np.add(conditionalProbs[pair[1]], pair[0])

		for line in conditionalProbs:
			s=np.sum(line)
			for i in range(len(line)):
				line[i]=(line[i]+1)/(s+2)

		return conditionalProbs

	def countMatrix(self, matrix):


		conditionalProbs=np.zeros([len(self.categories), len(self.Voc)])

		for pair in matrix:
			conditionalProbs[pair[1]]=np.add(conditionalProbs[pair[1]], pair[0])

		return conditionalProbs

	def getNmax(self, n, dictMax=False):

		countM=self.countMatrix(self.matrix)
		newVoc={}
		catMax={}
		index=0
		for c in self.categories.keys():
			temp={}
			temp2={}
			for word in self.Voc.keys():
				temp[word]=countM[self.categories[c]][self.Voc[word]]
			maxes=nlargest(n, temp, temp.get)
			for word in maxes:
				if word not in newVoc:
					newVoc[word]=index
					index+=1
				temp2[word]=temp[word]
			catMax[c]=temp2

		if dictMax==False:
			return newVoc
		else:
			return catMax
		

	def preTreatment(self, Voc, matrix):
		

		condProbs=self.countMatrix(matrix)
		var=np.var(condProbs[:, :], axis=0)
		
		newVoc={}
		median=np.mean(var)
		std=np.std(var)
		index=0
		for word in Voc.keys():
			if var[Voc[word]]<median+4*std:
				newVoc[word]=index
				index+=1

		return newVoc, self.genMatrix(self.trainSet, newVoc, self.categories)


	def getCatCount(self,trainSet, categories):
	
		priors=[0]*len(categories)

		for i in range(len(trainSet)):
			priors[categories[trainSet[i][1]]]+=1
		return priors

	def genVector(self,abstract, Voc):
		vector=[0]*len(Voc.keys())
		for word in abstract:
			if word in Voc.keys():
				vector[Voc[word]]+=1
		return vector


	def genMatrix(self,trainSet, Voc, categories):
		
		dataMatrix=[]

		for i in range(len(trainSet)):
			text=trainSet[i][0]
			cat=categories[trainSet[i][1]]
			vector=self.genVector(text, Voc)
			entry=[vector, cat]
			dataMatrix.append(entry)

		return dataMatrix

	def genTestMatrix(self, testSet, Voc, categories):
		dataMatrix=[]

		for i in range(len(testSet)):
			text=testSet[i][0]
			cat=categories[testSet[i][1]]
			vector=self.genVector(text, Voc)
			entry=[vector, cat]
			dataMatrix.append(entry)

		return dataMatrix

	def getCategoryFromI(self, index):

		for cat in self.categories.keys():
			if self.categories[cat]==index:
				return cat

	def computeGood(self, predictions):
		temp=0
		for i in range(len(self.testSet)):
			real=self.testSet[i][1]
			num=self.categories[real]
			if predictions[i]==num:
				temp+=1
		return temp/len(predictions)

	def compare(self, label, prediction):
		if prediction==label:
			return True
		else:
			return False

