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
