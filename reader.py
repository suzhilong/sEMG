# -*- coding: utf-8 -*-
import numpy as np
import re
import os 
import sys
reload( sys )
sys.setdefaultencoding('utf-8')

class FilesReader(object):
	'''
	Read data from txt files,every file named with Serial number of the gesture.
	this class can read all the files in a path.
	'''

	def __init__(self, columns, path = "defult" ):
		self.columns = columns
		if path == "defult":
			self.path = os.getcwd()
		else:
			self.path = path
		self.fileList = [] #list path下以".txt"结尾的文件列表
		self.fileNumber = 0	 #int path下的文件数量，也就是动作的数量
		self.filesLength = {} #dict 已导入文件的各自行数
		self._nameSortMethod = lambda x:int(x[:-4]) #把x除后4个字符的串取出来并转换成int，在这里时把".txt"文件除扩展名之外的文件名取出来，后面用来排序
		self._CreatFileList()
	def __repr__(self):
		return 'FilesReader(columns={0.columns!r},path={0.path!s})'.format(self)
		
	def _CreatFileList(self):
		OriFileList = os.listdir(self.path)
		self.fileList =[file for file in OriFileList if file.endswith(".txt")] 
		self.fileList.sort(key = self._nameSortMethod) #文件名排序
		self.fileNumber = len(self.fileList)
		for file in self.fileList:
			self.filesLength[file] = 0 #初始化
			
	@property
	def files(self):
		return self.fileList #排好序的.txt文件列表
		
	@property
	def allFilesData(self):
		matList = []
		for fileIndex, fileName in enumerate(self.fileList):
			matList.append(self.loadFile(fileName))
		return 	matList #一个文件内的全部数据合并成的列表
		
	def loadFile(self,fileName):
		file = open(self.path +'/'+fileName,'r')
		# match the data in file.read()
		matchList = re.findall(r'[0-9e.-]+',file.read())
		tempMat = np.array(matchList,dtype = 'float')
		# calculate the length of the file
		fileLength = len(tempMat)/self.columns
		# reshape the data
		dataMat = np.reshape(tempMat,(fileLength,self.columns))
		self.filesLength[fileName] = fileLength #保存着文件名长度的字典
		return dataMat #返回所有数据组成的矩阵