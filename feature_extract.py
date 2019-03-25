# -*- coding: utf-8 -*-
#运行之后能提取出txt里的特征值并保存到目标txt里
from tsfeature import feature_core
import numpy as np
import pandas as pd
import re

path = "/home/su/code/sEMG/3.14sEMG/code/data/rawData/"
for pose in range(1,2):
	fileName = "%s.txt"%pose
	######################
	#读进数据并变成矩阵
	reader = open(path+fileName,'r')
	matchList = re.findall(r'[0-9.-]+',reader.read())
	tempMat = np.array(matchList,dtype = 'float')
	dataMat = np.reshape(tempMat,(len(tempMat)/4,4))

	chanels = [1,2,3,4] #4个通道
	rawFram = pd.DataFrame(dataMat[:,:],columns=chanels)

	#提取每个通道的特征值
	feature_list = []
	for i in range(1,len(chanels)+1):
		channel_feature = feature_core.sequence_feature(rawFram[1],1000,1000) 
		feature_list.extend(channel_feature.tolist())
	#print feature_list[0][0]

	#把None替换成0
	#print feature_list[0][13]
	for irow in range(len(feature_list)):
		for jcolumn in range(len(feature_list[irow])):
			if feature_list[irow][jcolumn]==None:
				feature_list[irow][jcolumn]=0				
	#print feature_list[0][13]

	'''
	row = len(feature_list)/len(chanels)
	column = len(chanels)
	featureMat = np.zeros((row,column))
	l = 0
	for j in range(row):
		for k in range(column):
			featureMat[j][k] = feature_list[l]
			l += 1
	print featureMat.shape
	'''






	#####写数据,按照通道的顺序把特征值写到txt里,每一行是19个特征值
	featureTxt = 'feature%s.txt'%pose
	writer = open('/home/su/code/sEMG/3.14sEMG/code/data/features/'+featureTxt,'w')
	for line in feature_list:
		writer.writelines(str(line)+'\n')
	writer.close()