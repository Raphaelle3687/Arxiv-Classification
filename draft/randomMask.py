import numpy as np
import numpy.random as r
class Mask:

	def __init__(self,name, Voc, COVRG=0.1, dR=0.5, mask=[]):
		self.name=name
		self.Voc=Voc
		self.COVRG=COVRG
		self.deathRate=dR
		self.mask=mask
		self.gen=0

	def create(self):

		self.mask=np.zeros(len(self.Voc))

		self.N=int(self.COVRG*len(self.Voc))
		
		for i in range(len(self.Voc)):
			chance=r.random()
			if chance<self.COVRG:
				self.mask[i]=1

	def mutate(self):
		
		countLive=0
		countDeath=0
		newMask=np.zeros(len(self.mask))
		for i in range(len(self.mask)):
			if self.mask[i]==1:
				countLive+=1
				newMask[i]=1
			else:
				countDeath+=1


		p1=self.deathRate
		p0=(countLive*p1)/countDeath

		for i, val in enumerate(newMask):
			prob=r.random()
			if val==1:
				if prob<p1:
					newMask[i]=0
			elif val==0:
				if prob<p0:
					newMask[i]=1

		self.gen+=1

		return Mask(self.name+"." +str(self.gen), self.Voc, countLive/len(newMask), max((self.deathRate*0.95, 0.15)), newMask)


	def average(self, maskList):

		countLive=0
		newMask=np.zeros(len(self.Voc))

		for i in range(len(newMask)):
			temp=0
			for mask in maskList:				
				if mask.mask[i]==1:
					temp+=1 
			#if temp/len(maskList)>0.5:
			#	newMask[i]=1
			#	countLive+=1
			newMask[i]=temp/len(maskList)
		COVRG=countLive/len(newMask)
		dr=self.deathRate
		name="average;"+self.name
		return Mask(name, self.Voc, COVRG=COVRG,dR=dr, mask=newMask )


class Mask2:

	def __init__(self,name, Voc, var=0.5, mask=[]):


		self.name=name
		self.Voc=Voc
		self.var=var
		self.mask=mask
		self.gen=0

	def create(self):

		self.mask=self.var*r.randn(len(self.Voc))
		self.mask=self.mask+2

	def mutate(self):
		
		newMask=self.var*r.randn(len(self.mask))

		newMask=np.add(newMask, self.mask)

		self.gen+=1

		return Mask2(self.name+"." +str(self.gen), self.Voc, self.var, mask=newMask)


	def average(self, maskList):

		newMask=np.zeros(len(self.Voc))

		for mask in maskList:
			newMask=np.add(newMask, mask.mask)

		newMask=newMask/len(maskList)
		name="average;"+self.name
		return Mask2(name, self.Voc, self.var, mask=newMask)









"""p0*{0}+{1}-p1*{1}=cov
p0*{0}+q1*{1}=cov
p0=(cov-q1*{1})/{0}"""


