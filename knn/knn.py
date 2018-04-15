#encoding:utf8
import numpy as np
import glob
import os

#knn算法主体
def knn_classifier(x_train,y_train,x_test,k):
	#计算向量拘留
	dis = distance(x_train,x_test)
	#根据距离排序
	sort_index = dis.argsort()
	classCount = {}
	#取前k个邻近的点
	for i in range(k):
		#获取第i个点的类别
		label = y_train[sort_index[i]]
		classCount[label] = classCount.get(label,0) + 1
	#进行多数表决投票
	classCount = sorted(classCount.items(),lambda x,y:cmp(x[1],y[1]),reverse=True)
	return classCount[0][0]



def distance(x_train,x_test):
	datasize = x_train.shape[0]
	#tile可以把一个向量重复叠加为一个矩阵
	diff = np.tile(x_test,(datasize,1)) - x_train
	squareDiff = diff ** 2
	squareSum = squareDiff.sum(axis = 1)
	dis = np.sqrt(squareSum)
	return dis

#把手写体32*32的像素矩阵转化为1*2014的向量
def img2Vector(filename):
	returnVector = np.zeros((1,1024))
	file = open(filename)
	for i in range(32):
		lineString = file.readline()
		for j in range(32):
			returnVector[0,32 * i + j] = int(lineString[j])
	return returnVector


def load_train_data():
	train_data_file_path = glob.glob('./digits/trainingDigits/*.txt')
	train_label = []
	filenum = len(train_data_file_path)
	train_data = np.zeros((filenum,1024))
	for i in range(filenum):
		file_path = train_data_file_path[i]
		label = os.path.basename(file_path).split('_')[0]
		train_label.append(label)
		train_data[i:] = img2Vector(file_path)
	return train_data,train_label

def hand_writing_class_test():
	train_data,train_label = load_train_data()
	test_data_file_path = glob.glob('./digits/testDigits/*.txt')
	error = 0.0
	count = 0
	for file_path in test_data_file_path:
		file = open(file_path)
		test_label = os.path.basename(file.name).split('_')[0]
		test_data = img2Vector(file_path)
		predict_label = knn_classifier(train_data,train_label,test_data,3)
		count += 1
		if predict_label!=test_label:
			print "predict_label: ",predict_label,", test_label: ",test_label
			error += 1
	print 'error rate: ',error/count

def test_knn_classifier():
	x_train = np.array([[1,1.1],[1,1],[0,0],[0,0.1]]).astype(float)
	y_train = ['A','A','B','B']
	x_test = np.array([0,0]).astype(float)
	res = knn_classifier(x_train,y_train,[0,0],3)
	print res

def main():
	hand_writing_class_test()

if __name__=='__main__':
	main()