from numpy import *
import operator
import string

def classify0(inX, dataSet, labels, k):
    inx=array(inX)
    dataSetSize = dataSet.shape[0]
    diffMat0 = tile(inx,(dataSetSize,1))
    diffMat=diffMat0-dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    #
    classCount={}
    for i in range(k):
        voteIlabel = (str)(labels[sortedDistIndicies[i]])
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #
    return sortedClassCount[0][0]

def ReadFile2Mat0(fileName):
    fr=open(fileName)
    init=1
    for line in fr.readlines():
        str=line.strip('\n').split(',')
        t=array([str[0],str[1],str[2]]).astype(float)
        t1=array(str[3]).astype(float)

        if(init==1):
            ReturnMat=t
            labels=t1
            init=0
        else:
            ReturnMat=row_stack([ReturnMat,t])
            labels=row_stack([labels,t1])
    return ReturnMat,labels

def ReadFile2Mat(fileName):
    fr=open(fileName)
    ReturnMat=[]
    labels=[]
    for line in fr.readlines():
        str=line.strip('\n').split(',')
        t=array([str[0],str[1],str[2]]).astype(float)
        t1=array(str[3]).astype(float)

        ReturnMat.append(t)
        labels.append(t1)

    return ReturnMat,labels