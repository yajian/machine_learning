#encoding:utf8
class Node(object):
	def __init__(self,point,left,right,split,label):
		#左子树
		self.left = left
		#右子树
		self.right = right
		#数据节点
		self.point = point
		#分割维度
		self.split = split
		#标签
		self.label = label

	# def display(self,intent = 0):
	# 	print("(%d) %s %s"%(intent,' '*intent,str(self.point)))
	# 	if self.left != None:
	# 		self.left.display(intent+1)
	# 	if self.right != None:
	# 		self.right.display(intent+1)