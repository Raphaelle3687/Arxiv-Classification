import pandas
import numpy as np
import math
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
import string
from nltk.corpus import stopwords 
from heapq import nlargest

class reducedDataSet:
	
	def __init__(self,seed, N, catDict, mode=0):
		self.categoriesDict=catDict
		self.seed=seed%5
		self.N=N
		self.categories=None
		self.catCount=None
		self.bruteVoc=None
		self.Voc=None
		self.matrix=None
		self.trainSet=None
		self.validSet=None
		self.testSet=None
		self.cleanData=None
		if mode==0:
			self.initTest()



	def initTest(self):

		brute=self.getTrainData()
		categories=brute[1]
		print(categories)
		abstractSet=brute[0]
		cleaned=self.format(abstractSet)
		self.cleanData=cleaned

		trainSet,validSet, testSet=self.split_dataset(self.cleanData)

		self.trainSet=trainSet
		self.validSet=validSet
		self.testSet=testSet

		Voc=self.genVoc(trainSet, categories)
		self.catCount=self.getCatCount(trainSet, categories)		

		self.bruteVoc=Voc
		self.Voc=Voc
		print(len(self.bruteVoc))
		self.categories=categories

		self.matrix=self.genMatrix(trainSet, self.bruteVoc, categories)

		#self.reduceWords(self.N)


	def getMiniDataSet(self,categoryList, N):

		trainData=self.subsetData(categoryList, self.trainSet)
		miniCat={}
		for i in categoryList:
			miniCat[i]=len(miniCat)
		miniSet=miniDataSet(trainData, miniCat, self.bruteVoc, N)
		return miniSet

	def subsetData(self, categoryList, set):
		newData=[]
		for pair in set:
			if pair[1] in categoryList:
				newData.append(pair)

		return newData

	def reduceWords(self, n):
		newVoc=self.getNmax(n)
		self.Voc=newVoc
		print(len(self.Voc))
		self.matrix=self.genMatrix(self.trainSet, self.Voc, self.categories)


	def getTrainData(self):
		trainData=pandas.read_csv("train.csv")
		data=[trainData["Abstract"],trainData["Category"]]
		categories={}
		for i in data[1]:
			if self.categoriesDict[i] not in categories:
				categories[self.categoriesDict[i]]=len(categories)

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
				#carToClean=[":", ";", "."]
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
	            splitTest1.append(dataSet[i])
	        elif i%5==(self.seed+1%5):
	        	splitValid1.append(dataSet[i])
	        else:
	        	splitTrain1.append(dataSet[i])

	    return (splitTrain1,splitValid1, splitTest1)

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


	def probMatrix(self, matrix, Voc, categories):

		conditionalProbs=np.zeros([len(categories), len(Voc)])

		for pair in matrix:
			conditionalProbs[pair[1]]=np.add(conditionalProbs[pair[1]], pair[0])

		for line in conditionalProbs:
			s=np.sum(line)
			for i in range(len(line)):
				line[i]=(line[i]+1)/(s+2)

		return conditionalProbs

	def countMatrix(self, matrix, Voc, categories):


		conditionalProbs=np.zeros([len(categories), len(Voc)])

		for pair in matrix:
			conditionalProbs[pair[1]]=np.add(conditionalProbs[pair[1]], pair[0])

		return conditionalProbs

	def getNmax(self, n, dictMax=False):

		countM=self.countMatrix(self.matrix, self.bruteVoc, self.categories)
		newVoc={}
		catMax={}
		index=0
		for c in self.categories.keys():
			temp={}
			temp2={}
			for word in self.bruteVoc.keys():
				temp[word]=countM[self.categories[c]][self.bruteVoc[word]]
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


		

	def varTreatment(self, Voc, matrix, n):
		

		condProbs=self.countMatrix(matrix)
		var=np.var(condProbs[:, :], axis=0)
		
		newVoc={}
		median=np.mean(var)
		std=np.std(var)
		index=0
		for word in Voc.keys():
			if var[Voc[word]]<median+n*std:
				newVoc[word]=index
				index+=1

		self.Voc=newVoc
		Self.matrix=self.genMatrix(self.trainSet, newVoc, self.categories)


	def getCatCount(self,trainSet, categories):
	
		priors=[0]*len(categories)

		for i in range(len(trainSet)):
			priors[categories[self.categoriesDict[trainSet[i][1]]]]+=1
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
			cat=categories[self.categoriesDict[trainSet[i][1]]]
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
		if prediction==self.categoriesDict[label]:
			return True
		else:
			return False


class miniDataSet():


	def __init__(self,trainData, categories, Voc, N):
		self.categories=categories
		self.N=N
		self.Voc=Voc
		self.trainSet=trainData
		self.matrix=self.genMatrix(self.trainSet, self.Voc, self.categories)
		self.catCount=self.getCatCount(self.trainSet, self.categories)

		#self.varTreatment(self.Voc, self.matrix, N)

		#self.reduceWords(self.N)


	def reduceWords(self, n):
		newVoc=self.getNmax(n)
		self.Voc=newVoc
		print(len(self.Voc))
		self.matrix=self.genMatrix(self.trainSet, self.Voc, self.categories)

	def split_dataset(self,dataSet):
	    splitTrain1=[]
	    #splitValid1=[]
	    splitTest1=[]
	    for i in range(len(dataSet)):

	        if i%5==self.seed%5:
	            splitTest1.append(dataSet[i])
	        #elif i%5==(self.seed+1)%5:
	            #splitTest1.append(dataSet[i])
	        else:
	        	splitTrain1.append(dataSet[i])

	    return (splitTrain1, splitTest1)


	def probMatrix(self, matrix, Voc, categories):

		conditionalProbs=np.zeros([len(self.categories), len(self.Voc)])

		for pair in matrix:
			conditionalProbs[pair[1]]=np.add(conditionalProbs[pair[1]], pair[0])

		for line in conditionalProbs:
			s=np.sum(line)
			for i in range(len(line)):
				line[i]=(line[i]+1)/(s+2)

		return conditionalProbs

	def countMatrix(self, matrix, Voc, categories):


		conditionalProbs=np.zeros([len(categories), len(Voc)])

		for pair in matrix:
			conditionalProbs[pair[1]]=np.add(conditionalProbs[pair[1]], pair[0])

		return conditionalProbs

	def varTreatment(self, Voc, matrix, n):
		

		condProbs=self.countMatrix(matrix, Voc, self.categories)
		var=np.var(condProbs[:, :], axis=0)
		
		newVoc={}
		median=np.mean(var)
		std=np.std(var)
		index=0
		for word in Voc.keys():
			if var[Voc[word]]<median+n*std and var[Voc[word]]>median-n*std:
				newVoc[word]=index
				index+=1

		self.Voc=newVoc
		self.matrix=self.genMatrix(self.trainSet, newVoc, self.categories)

	def getNmax(self, n, dictMax=False):

		countM=self.countMatrix(self.matrix, self.Voc, self.categories)
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

	def getNotMax(self, n, dictMax=False):

		countM=self.countMatrix(self.matrix, self.Voc, self.categories)
		newVoc={}
		catMax={}
		index=0
		for c in self.categories.keys():
			temp={}
			temp2={}
			for word in self.Voc.keys():
				temp[word]=countM[self.categories[c]][self.Voc[word]]
			maxes=nlargest(n, temp, temp.get)
			for word in self.Voc:
				if word not in newVoc and word not in maxes:
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
		if prediction==self.categoriesDict[label]:
			return True
		else:
			return False

