from ..core import Node

class Add(Node):
    def compute(self):
        self.value = self.parent1.value + self.parent2.value
    def get_parent_grad(self, parent):
        return 1

class Mul(Node):
    def compute(self):
        self.value = self.parent1.value * self.parent2.value
    def get_parent_grad(self, parent):
        '''从parent1,2改成parents时，需要重写'''
        if parent == self.parent1:
            return self.parent2.value
        elif parent == self.parent2:
            return self.parent1.value
        else:
            raise "get which is not a parent of mul node"

class Step(Node):
    def compute(self):
        '''@阶跃函数。单个父节点的值作为输入，产生一个标量输出'''
        self.value = 1 if self.parent1.value > 0 else 0
    def get_parent_grad(self, parent):  # parent: 有多个父节点时，知道是求哪个父节点的导数，但Step仅一个父节点。
        '''problem --> 阶跃函数不可导，如何理解它的求导'''
        return 0. if self.parent1.value > 0 else -1.

