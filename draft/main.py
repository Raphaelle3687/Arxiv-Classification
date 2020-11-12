from dataSet import DataSet
from perceptron import PerceptronBasic
from sklearn import svm
from multiBayes import multinomialBayes
from reducedDataSet import reducedDataSet
import numpy as np
from layerBayes import layerBayes
import pandas
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
import string
from nltk.corpus import stopwords 
from heapq import nlargest
from randomMask import Mask
from randomMask import Mask2

def format(data):

	lem=WordNetLemmatizer()
	stem=PorterStemmer()
	newFormat=[]
	for i in range(len(data)):
		text = data[i].split(" ")
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



		newEntry=text4
		newFormat.append(newEntry)
					
	return newFormat

def mainPerc():
	data=DataSet(16)
	data.reduceWords(250)
	perc=PerceptronBasic(data)
	perc.train(10*len(data.trainSet))
	preds=perc.predictOnTest()
	print(perc.computeGood(preds))

def simpleSVM():

	data=DataSet(19)

	X=[]
	y=[]
	matrixTrain=data.matrix

	for pair in matrixTrain:
		X.append(pair[0])
		y.append(pair[1])

	matrixTest=data.genTestMatrix(data.testSet, data.Voc, data.categories)
	Xtest=[]

	for pair in matrixTest:
		Xtest.append(pair[0])


	clf=svm.SVC()
	clf.fit(X, y)
	pred=clf.predict(Xtest)

	print(data.computeGood(pred))


def findBestMask2(data1, validSet, testSet, catDict=None):

	#data1.reduceWords(1000)

	nBayes=50
	generations=10
	maskBayes={}
	maskBayes2={}
	name1=0
	bayes1=multinomialBayes(data1)


	valX=[]
	valY=[]
	for pair in validSet:
		valX.append(pair[0])
		if catDict==None:
			valY.append(pair[1])
		else:
			valY.append(catDict[pair[1]])

	preds=bayes1.predictTest(valX)
	print(computeGood(preds, valY, simple=True))
	
	for i in range(nBayes):
		name1+=1
		m=[]
		for cat in data1.categories:
			mTemp=Mask("mask."+str(name1), data1.Voc)
			mTemp.create()
			m.append(mTemp)
		maskBayes2["mask."+str(name1)]=m

	for gen in range(generations):
		maskBayes=maskBayes2
		print(str(gen)+"-------------------")
		maskScores={}

		for mask in maskBayes.keys():
			bayes1.mask=maskBayes[mask]
			#print(maskBayes[mask])
			preds=bayes1.predictTest(valX)
			maskScores[mask]=computeGood(preds, valY, simple=True)
			print(mask, maskScores[mask])

		reprod=5
		maxes=nlargest(int(nBayes/reprod), maskScores, maskScores.get)

		maskBayes2={}

		for maxMask in maxes:
			maskBayes2[maxMask]=maskBayes[maxMask]
			for i in range(reprod-1):
				newM=[]
				for cat in data1.categories:
					new=maskBayes[maxMask][data1.categories[cat]].mutate()
					newM.append(new)
				maskBayes2[new.name]=newM

	bestName=nlargest(10, maskScores, maskScores.get)
	bestMask=maskBayes[bestName[0]]
	#print(bestMask.name, bestMask.COVRG)

	bestList=[]
	for cat in data1.categories:
		temp=[]
		for nam in bestName:
			temp.append((maskBayes[nam][data1.categories[cat]]))
		bestList.append(temp)

	averageBest=[]
	for cat in data1.categories:
		averageBest.append(bestMask[data1.categories[cat]].average(bestList[data1.categories[cat]]))
	#print(averageBest.name, averageBest.COVRG)

	testX=[]
	testY=[]
	for pair in testSet:
		testX.append(pair[0])
		if catDict==None:
			testY.append(pair[1])
		else:
			testY.append(catDict[pair[1]])

	bayes1.mask=None

	preds=bayes1.predictTest(testX)
	print(computeGood(preds, testY))

	bayes1.mask=averageBest

	preds=bayes1.predictTest(testX)
	print(computeGood(preds, testY))


def findBestMask(data1, validSet, testSet, catDict=None):

	#data1.reduceWords(3000)

	nBayes=120
	generations=8
	maskBayes={}
	maskBayes2={}
	name1=0
	bayes1=multinomialBayes(data1)


	valX=[]
	valY=[]
	for pair in validSet:
		valX.append(pair[0])
		if catDict==None:
			valY.append(pair[1])
		else:
			valY.append(catDict[pair[1]])

	preds=bayes1.predictTest(valX)
	print(computeGood(preds, valY, simple=True))
	
	for i in range(nBayes):
		name1+=1
		m=Mask("mask."+str(name1), data1.Voc)
		m.create()
		maskBayes2["mask."+str(name1)]=m

	for gen in range(generations):
		maskBayes=maskBayes2
		print(str(gen)+"-------------------")
		maskScores={}

		for mask in maskBayes.keys():
			bayes1.mask=maskBayes[mask]
			preds=bayes1.predictTest(valX)
			maskScores[mask]=computeGood(preds, valY, simple=True)
			print(mask, maskScores[mask])

		reprod=6
		maxes=nlargest(int(nBayes/reprod), maskScores, maskScores.get)

		maskBayes2={}

		for maxMask in maxes:
			maskBayes2[maxMask]=maskBayes[maxMask]
			for i in range(reprod-1):
				new=maskBayes[maxMask].mutate()
				maskBayes2[new.name]=new

	bestName=nlargest(10, maskScores, maskScores.get)
	bestMask=maskBayes[bestName[0]]
	print(bestMask.name, bestMask.COVRG)

	bestList=[]
	for nam in bestName:
		bestList.append(maskBayes[nam])

	averageBest=bestMask.average(bestList)
	print(averageBest.name, averageBest.COVRG)

	testX=[]
	testY=[]
	for pair in testSet:
		testX.append(pair[0])
		if catDict==None:
			testY.append(pair[1])
		else:
			testY.append(catDict[pair[1]])

	bayes1.mask=None

	preds=bayes1.predictTest(testX)
	print(computeGood(preds, testY))

	bayes1.mask=averageBest

	preds=bayes1.predictTest(testX)
	print(computeGood(preds, testY))




def maskBayes():

	data1=DataSet(16)
	#data2=DataSet(17)
	#data3=DataSet(18)
	data1.reduceWords(3000)
	#data2.reduceWords(3000)
	#data3.reduceWords(3000)


	nBayes=64
	generations=4
	maskBayes={}
	name1=0
	bayes1=multinomialBayes(data1)
	#bayes2=multinomialBayes(data2)
	#bayes3=multinomialBayes(data3)
	print(bayes1.predictOnTest())
	
	for i in range(nBayes):
		name1+=1
		m=Mask("mask."+str(name1), data1.Voc)
		m.create()
		maskBayes["mask."+str(name1)]=m

	for gen in range(generations):
		print(str(gen)+"-------------------")
		maskScores={}

		for mask in maskBayes.keys():
			bayes1.mask=maskBayes[mask]
			temp=bayes1.predictOnTest()
			maskScores[mask]=temp
			print(mask, maskScores[mask])

		reprod=8
		maxes=nlargest(int(nBayes/reprod), maskScores, maskScores.get)

		maskBayes2={}

		for maxMask in maxes:
			maskBayes2[maxMask]=maskBayes[maxMask]
			for i in range(reprod-1):
				new=maskBayes[maxMask].mutate()
				maskBayes2[new.name]=new

		maskBayes=maskBayes2

	bestName=nlargest(1, maskScores, maskScores.get)
	bestMask=maskBayes[bestName[0]]

	print(bestMask.name, bestMask.COVRG)


	#data4=DataSet(19)
	#bayes4=multinomialBayes(data4)

	testXX=[]
	testYY=[]
	for pair in data1.validSet:
		testXX.append(pair[0])
		testYY.append(pair[1])

	preds=bayes1.predictTest(testXX)
	print(computeGood(preds, testYY))
			



def bayes():
	data=DataSet(16)
	#data.reduceWords(500)
	bayes=multinomialBayes(data)
	print(bayes.predictOnTest())

def testCount():
	data=DataSet(16)
	n=500
	maxWords=data.getNmax(n, dictMax=True)
	overlap=[]
	read=[]
	for cat in data.categories:
		read.append(cat)
		for cat2 in data.categories:
			if cat2 in read:
				continue
			temp=0
			for word in maxWords[cat]:
				if word in maxWords[cat2]:
					temp+=1
			overlap.append([temp/n, cat+" "+cat2])

	sorted=False
	while not sorted:
		sorted=True
		for i in range(len(overlap)-1):
			if overlap[i][0]>overlap[i+1][0]:
				sorted=False
				temp=overlap[i]
				overlap[i]=overlap[i+1]
				overlap[i+1]=temp
	print(overlap)


def computeGood(predictions, labels, simple=False):
	good={}
	nLab={}

	for lab in labels:
		if lab not in good:
			good[lab]={}
			nLab[lab]=0

	for lab in good.keys():
		for lab2 in good.keys():
			good[lab][lab2]=0

	for i in range(len(predictions)):
		good[labels[i]][predictions[i]]+=1
		nLab[labels[i]]+=1

	s=0
	for lab in good.keys():
		for lab2 in good[lab].keys():
			good[lab][lab2]=good[lab][lab2]/nLab[lab]
		s+=good[lab][lab]

	good["Average"]=s/len(good)

	if simple==False:
		return good
	elif simple==True:
		return good["Average"]

def testMask():
	catDict={"astro-ph": "ASTRO", "astro-ph.CO":"ASTRO", "astro-ph.SR":"ASTRO", "astro-ph.GA":"ASTRO",
	"cs.LG":"CS", "stat.ML":"CS", 
	"math.CO":"MATH", "math.AP":"MATH",
	"physics.optics":"COND", "quant-ph":"COND", "cond-mat.mtrl-sci":"COND", "cond-mat.mes-hall":"COND",
	"gr-qc":"RELAT", "hep-th":"RELAT", "hep-ph":"RELAT"}

	masterClass={}
	for key in catDict.keys():
		if catDict[key] not in masterClass:
			masterClass[catDict[key]]=[key]
		else:
			masterClass[catDict[key]].append(key)

	data=reducedDataSet(16, 10000, catDict)

	astroData=data.getMiniDataSet(masterClass["ASTRO"], 60)
	astroValid=data.subsetData(masterClass["ASTRO"], data.validSet)
	astroTest=data.subsetData(masterClass["ASTRO"], data.testSet)		

	findBestMask2(astroData, astroValid, astroTest, catDict=None)
	#findBestMask2(data, data.validSet, data.testSet, catDict=catDict)



def reducedClassifier():
	catDict={"astro-ph": "ASTRO", "astro-ph.CO":"ASTRO", "astro-ph.SR":"ASTRO", "astro-ph.GA":"ASTRO",
	"cs.LG":"CS", "stat.ML":"CS", 
	"math.CO":"MATH", "math.AP":"MATH",
	"physics.optics":"COND", "quant-ph":"COND", "cond-mat.mtrl-sci":"COND", "cond-mat.mes-hall":"COND",
	"gr-qc":"RELAT", "hep-th":"RELAT", "hep-ph":"RELAT"}




	masterClass={}
	for key in catDict.keys():
		if catDict[key] not in masterClass:
			masterClass[catDict[key]]=[key]
		else:
			masterClass[catDict[key]].append(key)



	data=reducedDataSet(16, 10000, catDict)

	testXX=[]
	testYY=[]
	for pair in data.testSet:
		testXX.append(pair[0])
		testYY.append(pair[1])

	lBayes=layerBayes(data)
	preds=lBayes.predictTest(testXX, testYY)
	print("lBayes ",computeGood(preds, testYY))


	testXG=[]
	testYG=[]
	for pair in data.testSet:
		testXG.append(pair[0])
		testYG.append(catDict[pair[1]])

	print("MASTERCLASS")
	bayes=multinomialBayes(data)
	preds=bayes.predictTest(testXG)
	print("bayes ",computeGood(preds, testYG))

	miniTest={}
	testX={}
	testY={}

	for key in masterClass:
		miniTemp=data.subsetData(masterClass[key], data.testSet)		
		miniTest[key]=miniTemp
		testX[key]=[]
		testY[key]=[]
		for pair in miniTest[key]:
			testX[key].append(pair[0])
			testY[key].append(pair[1])

	for key in masterClass:
		print(key)
		miniData=data.getMiniDataSet(masterClass[key], 60)
		
		bayes=multinomialBayes(miniData)
		preds=bayes.predictTest(testX[key])
		print("bayes ",computeGood(preds, testY[key]))


def getCleanedTest():
	temp=getTestData()
	abstractTest=temp[0]
	cleanedTest=format(abstractTest)
	testSet=cleanedTest
	return testSet

def getTestData():
	trainData=pandas.read_csv("test.csv")
	data=[trainData["Abstract"]]

	return (data)

def predictionsToCSV(preds):
	ids=[]
	category=[]
	for i in range(len(preds)):
		ids.append(i)
		category.append(preds[i])
	dik={}
	dik["Category"]=preds
	frame=pandas.DataFrame.from_dict(dik)
	frame.to_csv("predictions", header=True)

def testBayes():

	data=DataSet(16, mode=1)
	bayes=multinomialBayes(data)
	testData=getCleanedTest()
	print(testData)

	predictions=bayes.predictTest(testData)
	predictionsToCSV(predictions)


	

#testBayes()
#reducedClassifier()


#testCount()
testMask()
#maskBayes()
#mainPerc()