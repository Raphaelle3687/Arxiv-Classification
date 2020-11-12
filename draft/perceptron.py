import pandas
import numpy as np
import math

class PerceptronBasic:

	def __init__(self, dataSet):

		self.dataSet=dataSet
		self.W=np.zeros([len(self.dataSet.categories), len(self.dataSet.Voc)])
		self.bias=[]
		self.initWb()


	def initWb(self):

		for i in self.dataSet.categories.keys():
			wi=np.zeros(len(self.dataSet.Voc))
			self.W[self.dataSet.categories[i]]=wi

		self.bias=np.zeros(len(self.dataSet.categories))



	def predict(self,abstractVector):
		#print(self.W, abstractVector)
		predictions=np.dot(self.W, abstractVector)
		predictions=np.add(predictions, self.bias)
		pred=np.argmax(predictions)
		return pred


	def train(self, itter):
		i=0
		while i<itter:
			pair=self.dataSet.matrix[i%(len(self.dataSet.matrix))]
			abstract=pair[0]
			category=pair[1]
			#print(category)
			pred=self.predict(abstract)
			if pred!=category:
				self.W[category]=np.add(self.W[category], abstract)
				self.bias[category]+=1
				self.W[pred]=np.subtract(self.W[pred], abstract)
				self.bias[pred]-=1
			i+=1


	def predictOnTest(self):
		predictions=[]
		for pair in self.dataSet.testSet:
			abstract=pair[0]
			abstractVector=self.dataSet.genVector(abstract, self.dataSet.Voc)
			pred=self.predict(abstractVector)
			predictedCat=self.dataSet.getCategoryFromI(pred)
			predictions.append(predictedCat)
		return predictions


	def predictTest(self, testSet):

		predictions=[]
		temp=0
		for abstract in testSet:
			abstractV=self.dataSet.genVector(abstract, self.dataSet.Voc)
			pred=self.predict(abstractV)
			textPred=self.dataSet.getCategoryFromI(pred)
			predictions.append(textPred)
		return predictions

	def computeGood(self, predictions):
		temp=0
		for i in range(len(self.dataSet.testSet)):
			real=self.dataSet.testSet[i][1]
			if predictions[i]==real:
				temp+=1
		return temp/len(predictions)









