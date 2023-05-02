class Node:
    def __init__(self, parent1=None, parent2=None) -> None:
        self.parent1 = parent1
        self.parent2 = parent2
        self.value = None

        self.grad = None  # 在其它结点求梯度时可能再次用到本结点的梯度
        self.children = []

        parents = [self.parent1, self.parent2]
        for parent in parents:
            if parent is not None:
                parent.children.append(self)

    def set_value(self, value):
        self.value = value
    def compute(self):
        '''抽象方法，在具体的类中重写'''
        pass
    def forward(self):
        for parent in [self.parent1, self.parent2]:
            if parent is not None and parent.value is None:
                parent.forward()
        self.compute()
        return self.value
    def get_parent_grad(self, parent):
        '''求本节点对于父节点的梯度，抽象方法'''
        pass
    def get_grad(self):
        '''求结果节点对本节点的梯度'''
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
        '''递归清除父节点的值和梯度信息'''
        self.grad = None
        if self.parent1 is not None:  # 清空非变量节点的值
            self.value = None
        for parent in [self.parent1, self.parent2]:
            if parent is not None:
                parent.clear()
    def update(self, lr=0.001):
        '''根据本节点的梯度，更新本节点的值'''
        self.value -= lr * self.grad  # 减号表示梯度的反方向



class Varrible(Node):
    def __init__(self) -> None:
        super().__init__()