from numpy import *

def standRegres(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

def rssError(yArr,yHatArr):
    X=mat(yArr)
    Y=mat(yHatArr)
    #return (((yArr-yHatArr)/yArr)**2).sum()
    return cov(X,Y)

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

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