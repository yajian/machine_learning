# coding=utf-8
import uuid


class TreeNode(object):
    def __init__(self, split_feat, split_value):
        self.uid = uuid.uuid1()
        self.split_value = split_value
        self.split_feat = split_feat
        self.left = None
        self.right = None
        self.weight = None
        self.isLeaf = False

    def get_weight(self, data):
        if self.isLeaf == True:
            return self.weight
        if data[:, self.split_feat] < self.split_value:
            return self.left.get_weight(data)
        else:
            return self.right.get_weight(data)

    def __str__(self):
        if self.isLeaf:
            return 'leaf: weight {}'.format(self.weight)
        else:
            return 'split feature:{}, split value:{}, left:[{}], right:[{}]'.format(self.split_feat, self.split_value,
                                                                                    self.left, self.right)
