# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import codecs
import numpy as np
import re

path = "/home/su/code/sEMG/3.14sEMG/code/data/rawData/"
fileName = "1.txt"
######################
#读进数据并变成矩阵
reader = open(path+fileName,'r')
matchList = re.findall(r'[0-9.-]+',reader.read())
tempMat = np.array(matchList,dtype = 'float')
dataMat = np.reshape(tempMat,(len(tempMat)/4,4))
####################
#画图
cut = [0,10000]
sliceData = dataMat[cut[0]:cut[1],:]
ChanList = ['Raw-c1','Raw-c2','Raw-c3','Raw-c4']
rawFrame = pd.DataFrame(sliceData, columns = ChanList)
plt.figure(figsize=(cut[1]/5000*1.5,len(ChanList)*2.5))#figsize=(width,high)
#画原始数据的图
for i in range(len(ChanList)):
	plt.subplot(4,1,1+i)
	plt.plot(rawFrame[ChanList[i]],ls = '-',lw = 1)
	plt.xlabel('time')
	plt.ylabel(ChanList[i])
plt.show()