from numpy import *
from math import *
#
# 预处理函数
#
def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

def regularize(xMat):
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    for i in range(shape(xVar)[1]):
        if xVar[:,i]==0: xVar[:,i]=1
    reMat = (xMat - xMeans) / xVar
    return reMat
#
# 算法函数
#
# -----------------------
# Stander Regres
# -----------------------
def standRegres(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws
# -----------------------
# LWLR
# -----------------------
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        # print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = testArr[i]*lwlr(testArr[i],xArr,yArr,k)
    return yHat

def lwlrMethod(dataArr,xTrainArr,yTrainArr,k=1.0):
    m = shape(dataArr)[0]
    m_t=shape(yTrainArr)[0]
    # First choose the best ws, using training data
    bestRss=inf;curRss=0;bestk=k
    for j in range(10):
        yHatT = zeros(m_t)
        try:
            for i in range(m_t):
                kn=k/10*j
                yHatT[i] = dataArr[i] * lwlr(yTrainArr[i], xTrainArr,yTrainArr,kn)
            curRss=(yTrainArr,yHatT)
            if curRss<bestRss:
                bestk=kn
                bestRss=curRss
        except: continue
    # Using the best ws to get the predict output
    yHat = zeros(m)
    yMax=max(yTrainArr);yMin=min(yTrainArr)
    for i in range(m):
        yHat[i] = dataArr[i] * lwlr(dataArr[i], xTrainArr, yTrainArr, bestk)
        _max=min(i+10,m);_min=max(0,m-20)
        yMax = max(yTrainArr);
        yMin = min(yTrainArr)
        try:
            yHat[i]=min(yHat[i],yMax*1.2)
            yHat[i]=max(yHat[i],yMin*0.8)
        except: continue
    return yHat
# -----------------------
# Ridge Regres
# -----------------------
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws
def ridgeTest(trainXArr,trainYArr,testXArr):
    xMat = mat(trainXArr); yMat=mat(trainYArr).T
    txMat=mat(testXArr)
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMeans = mean(xMat,0)
    xVar = var(xMat,0)
    xMat = (xMat - xMeans)/xVar
    txMeans = mean(txMat, 0)
    txVar = var(txMat, 0)
    txMat = (txMat - txMeans) / txVar
    # numTestPts = 30
    # wMat = zeros((numTestPts,shape(xMat)[1]))
    # re=zeros((numTestPts,shape(xMat)[0]))
    # for i in range(numTestPts):
    #     ws = ridgeRegres(xMat,yMat,exp(i-10))
    #     wMat[i,:]=ws.T
    #     re[i,:]=(xMat*ws+yMean).T   #Return predict YArr
    ws = ridgeRegres(xMat, yMat, exp(-10))
    re = zeros((1, shape(xMat)[0]))
    re[0,:] = (txMat * ws + yMean).T  # Return predict YArr
    return re

def stageWise(xArr,yArr,eps=1,numIt=1000):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    # yMat=regularize(yMat)
    m,n=shape(xMat)
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    returnMat = zeros((1, n))
    lowestError = inf;
    for i in range(numIt):
        # print(ws.T)
        for j in range(n):
            for sign in [-1,1]:
                wsTest = wsMax.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                # yVar=cov(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax[j] = wsTest[j]
        ws = wsMax.copy()
        returnMat=ws.T
    return returnMat