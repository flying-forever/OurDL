import random

class Graph:
    def __init__(self) -> None:
        self.nodes = []  # 在创建节点时自动加入
        self.nodes_in = []  # 也许可以改成定义一个方法，返回的是nodes的一个子集
        self.nodes_out = []
        self.node_label = None
        self.node_loss = None
    def add_node(self, node):
        '''
        - 在创建节点中自动调用，将节点加入计算图\n
        - 输入节点一定是变量节点，输出节点一定是运算节点（损失节点也需要前向传播？)
        - is_out参数在ops和loss中重复实现了
        '''
        self.nodes.append(node)
    def init_params(self):
        '''初始化可训练节点，使用[-0.5, 0.5]的均匀分布'''
        for node in self.nodes:  # 定义方法 --> 返回一个可训练节点的列表？
            if node.train:
                node.set_value(random.uniform(-0.5, 0.5))
    def set_nodes(self, nodes_in:list, nodes_out:list, node_loss:object, node_label:object):
        '''设置计算图中的特殊节点'''
        self.nodes_in = nodes_in
        self.nodes_out = nodes_out
        self.node_loss = node_loss
        self.node_label = node_label
    def forward(self, x:list, label):
        '''
        前向传播，到损失节点，并返回损失值 --> backward搭配使用
        @ problem
        1. label会是数组吗？
        '''
        for i, x_ in enumerate(x):
            self.nodes_in[i].set_value(x_)
        self.node_label.set_value(label)
        return self.node_loss.forward()
    def backward(self):
        '''计算所有节点的梯度，并更新可训练节点的值'''
        for node in self.nodes:
            if node.train:
                node.get_grad()
                node.update()
    def clear(self):
        '''调用损失节点的clear方法，递归清除计算图所有节点的梯度，和非变量节点（有上游节点）的值'''
        self.node_loss.clear()
    def predict(self, x:list):
        '''前向计算，返回结果节点的值'''
        self.node_loss.clear()
        for i, x_ in enumerate(x):
            self.nodes_in[i].set_value(x_)
        return [node.forward() for node in self.nodes_out]

# 全局对象
default_graph = Graph()
