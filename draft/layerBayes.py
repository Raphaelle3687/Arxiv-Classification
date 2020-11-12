import pandas
import numpy as np
import math
from multiBayes import multinomialBayes
class layerBayes:

	def __init__(self, reducedDataSet):

		self.data=reducedDataSet
		self.catDict=reducedDataSet.categoriesDict
		
		self.masterClass={}
		self.masterIndexes=self.data.categories
		
		for key in self.catDict.keys():
			if self.catDict[key] not in self.masterClass:
				self.masterClass[self.catDict[key]]=[key]
			else:
				self.masterClass[self.catDict[key]].append(key)

		sets={}
		self.layerIndexes=[0]*len(self.masterIndexes)

		for key in self.masterClass.keys():
			sets[key]=self.data.getMiniDataSet(self.masterClass[key], 60)
			self.layerIndexes[self.masterIndexes[key]]=sets[key].categories

		print(self.masterIndexes)
		print(self.layerIndexes)

		self.bayes1=multinomialBayes(self.data)

		self.bayesLayer2={}
		for key in self.masterClass.keys():
			self.bayesLayer2[key]=multinomialBayes(sets[key])

	def predict(self, abstract):

		masterScores=self.bayes1.predictLayer(abstract)
		layerScore=[0]*len(masterScores)

		for key in self.bayesLayer2.keys():
			score=self.bayesLayer2[key].predictLayer(abstract)
			layerScore[self.masterIndexes[key]]=score


		indexClass=masterScores.index(max(masterScores))
		clas=None
		for i in self.masterIndexes.keys():
			if self.masterIndexes[i]==indexClass:
				clas=i

		scoreTemp=layerScore[indexClass]
		ind=scoreTemp.index(max(scoreTemp))
		pred=None
		for i in self.layerIndexes[indexClass].keys():
			if self.layerIndexes[indexClass][i]==ind:
				pred=i

		return pred

	def predictTest(self, testSet, answers):

		predictions=[]
		temp=0
		for i in range(len(testSet)):
			pred=self.predict(testSet[i])
			predictions.append(pred)
		return predictions







