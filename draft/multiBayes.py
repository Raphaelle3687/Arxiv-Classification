import pandas
import numpy as np
import math

class multinomialBayes:

	def __init__(self, dataSet):
		self.mask=None
		self.dataSet=dataSet
		self.conditionalProbs=self.formProbs()
		self.priors=self.computePriors()

	def formProbs(self):

		matrix=self.dataSet.matrix
		conditionalProbs=np.zeros([len(self.dataSet.categories), len(self.dataSet.Voc)])

		for pair in matrix:
			conditionalProbs[pair[1]]=np.add(conditionalProbs[pair[1]], pair[0])

		for line in conditionalProbs:
			s=np.sum(line)
			for i in range(len(line)):
				line[i]=(line[i]+1)/(s+len(line))

		return conditionalProbs

	def computePriors(self):

		total=len(self.dataSet.trainSet)
		priors=[]
		for i in self.dataSet.catCount:
			priors.append(i/total)

		return priors


	def predict2(self, abstract):

		voc=self.dataSet.Voc
		cat=self.dataSet.categories
		
		score=[0]*len(cat)
		
		for word in abstract:
			if word in voc.keys():
				j=voc[word]
				
				if not (self.mask!=None and self.mask.mask[j]==1):

					for c in cat.keys():
						score[cat[c]]+=math.log(self.conditionalProbs[cat[c]][j])	


		for c in cat.keys():
			score[cat[c]]+=self.priors[cat[c]]

		return score.index(max(score))

	def activation(self, val, scale, mean):

		return 1/(1+math.exp(-scale*(val-mean)))

	def predict(self, abstract):

		voc=self.dataSet.Voc
		cat=self.dataSet.categories
		
		score=[0]*len(cat)
		
		for word in abstract:
			if word in voc.keys():
				j=voc[word]
				
				if self.mask==None:

					for c in cat.keys():
						score[cat[c]]+=math.log(self.conditionalProbs[cat[c]][j])	

				else:
					for c in cat.keys():
						val=self.conditionalProbs[cat[c]][j]
						thres=self.mask[cat[c]].mask[j]
						coeff=self.activation(thres, -10, 0.5)
						score[cat[c]]+=coeff*math.log(val)


		for c in cat.keys():
			score[cat[c]]+=self.priors[cat[c]]

		return score.index(max(score))

	def predictLayer(self, abstract):

		voc=self.dataSet.Voc
		cat=self.dataSet.categories
		
		score=[0]*len(cat)
		
		for word in abstract:

			if word in voc.keys():
			
				j=voc[word]
			
				for c in cat.keys():
					score[cat[c]]+=math.log(self.conditionalProbs[cat[c]][j])

		for c in cat.keys():
			score[cat[c]]+=math.log(self.priors[cat[c]])

		return score

	def predictTest(self, testSet):

		predictions=[]
		temp=0
		for abstract in testSet:
			pred=self.predict(abstract)
			textPred=self.dataSet.getCategoryFromI(pred)
			predictions.append(textPred)
		return predictions



	def predictOnTest(self):
		predictions=[]
		temp=0
		for pair in self.dataSet.testSet:
			pred=self.predict(pair[0])
			textPred=self.dataSet.getCategoryFromI(pred)
			predictions.append(textPred)
			if self.dataSet.compare(pair[1], textPred):
				temp+=1
		return temp/len(predictions)

"""for word in voc.keys():
			j=voc[word]
			for c in cat.keys():				
				p=self.conditionalProbs[cat[c]][j]
				if word in abstract:
					score[cat[c]]+=math.log(p)
				else:
					score[cat[c]]+=math.log(1-p)"""


