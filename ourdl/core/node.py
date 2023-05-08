from .graph import default_graph

class Node:
    def __init__(self, parents=[], parent1=None, parent2=None, train:bool=False) -> None:
        # say --> 兼容以前使用parent1、parent2参数创建结点的用法
        ''' 
        @ problem\n
        1. 要实现节点的“名字”属性吗？\n
        @ 可能的参数传递情况
        1 (parents=p1)
        2 (parents=p1, parent1=p2)
        3 (parents=[], parent1=p1, parent2=p2)
        4 (parents=[p1, p2, p3, ...])
        '''
        if parents and type(parents) == list:
            self.parents = parents
        else:
            self.parents = []
            temp_parents = [parents, parent1, parent2]
            for p in temp_parents:
                if p:
                    self.parents.append(p)
        self.children = []
        for parent in self.parents:
            parent.children.append(self)  # problem --> 好像有点问题
        
        self.train = train
        self.value = None
        self.grad = None  # 在其它结点求梯度时可能再次用到本结点的梯度
        self.graph = default_graph  # 待修改 --> 从参数列表指定节点所属的计算图
        self.graph.add_node(self)  # 将本节点添加到计算图中
        
    def set_value(self, value):
        self.value = value
    def compute(self):
        '''
        抽象方法，在具体的类中重写'''
        pass
    def forward(self):
        for parent in self.parents:
            if parent is not None and parent.value is None:
                parent.forward()
        self.compute()
        return self.value
    def get_parent_grad(self, parent):
        '''
        求本节点对于父节点的梯度，抽象方法'''
        pass
    def get_grad(self):
        '''
        求结果节点对本节点的梯度'''
        # 结果结点返回单位值，而不是self.value
        if not self.children:
            return 1
        if self.grad is not None:
            return self.grad
        else:
            self.grad = 0
            for i in range(len(self.children)):
                grad1 = self.children[i].get_parent_grad(parent=self)  # 子节点对自己的梯度
                grad2 = self.children[i].get_grad()  # 结果节点对子节点的梯度
                self.grad += grad1 * grad2
            return self.grad
    def clear(self):
        '''
        递归清除本节点和父节点的值和梯度信息'''
        self.grad = None
        if self.parents:
            self.value = None  # 清空非变量节点的值
        for parent in self.parents:
            parent.clear()
    def update(self, lr=0.01):
        '''
        根据本节点的梯度，更新本节点的值'''
        self.value -= lr * self.grad  # 减号表示梯度的反方向

class Varrible(Node):
    '''
    就和Node类一样'''
    pass
