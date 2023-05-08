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
        '''
        初始化可训练节点，使用[-0.5, 0.5]的均匀分布'''
        for node in self.nodes:  # 定义方法 --> 返回一个可训练节点的列表？
            if node.train:
                node.set_value(random.uniform(-0.5, 0.5))
    def backward(self):
        '''
        计算所有节点的梯度，并更新可训练节点的值'''
        for node in self.nodes:
            if node.train:
                node.get_grad()
                node.update()
    def forward(self):
        '''
        计算图的前向计算，仅到结果节点'''
        return [node.forward() for node in self.nodes_out]

    
# 全局对象
default_graph = Graph()
