from ..core import Node

class Add(Node):
    def compute(self):
        self.value = 0
        for parent in self.parents:
            self.value += parent.value

    def get_parent_grad(self, parent):
        return 1

class Mul(Node):
    '''仅支持两个父节点'''
    def compute(self):
        assert len(self.parents) == 2
        self.value = self.parents[0].value * self.parents[1].value
    def get_parent_grad(self, parent):
        if parent == self.parents[0]:
            return self.parents[1].value
        else:
            return self.parents[0].value

class Step(Node):
    def compute(self):
        '''
        @阶跃函数。单个父节点的值作为输入，产生一个标量输出
        暂时仅支持一个父节点'''
        assert len(self.parents) == 1
        self.value = 1 if self.parents[0].value > 0 else 0
    def get_parent_grad(self, parent):  # parent: 有多个父节点时，知道是求哪个父节点的导数，但Step仅一个父节点。
        '''problem --> 阶跃函数不可导，如何理解它的求导?'''
        return 0. if self.parents[0].value > 0 else -1.


class Relu(Node):
    def compute(self):
        assert len(self.parents) == 1
        self.value = self.parents[0].value if self.parents[0].value >= 0 else 0
    def get_parent_grad(self, parent):
        return 1. if self.parents[0].value > 0 else 0  # 发现relu的导函数就是step
    @staticmethod
    def relu(x: float):
        '''静态方法 --> 在计算图之外使用relu'''
        return x if x >= 0. else 0.


class LeakyRelu(Node):
    '''消除了relu中导数为0的情况'''
    def compute(self):
        assert len(self.parents) == 1
        t = self.parents[0].value
        self.value = t if t >= 0 else t * 0.1
    def get_parent_grad(self, parent):
        return 1. if self.parents[0].value > 0 else 0.1  # 发现relu的导函数就是step
    @staticmethod
    def relu(x: float):
        '''静态方法 --> 在计算图之外使用leakyrelu'''
        return x if x >= 0. else x * 0.1
