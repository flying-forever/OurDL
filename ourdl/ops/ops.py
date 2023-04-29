from ..core import Node

class Add(Node):
    def __init__(self, parent1=None, parent2=None) -> None:
        super().__init__(parent1, parent2)
    def compute(self):
        self.value = self.parent1.value + self.parent2.value
    def get_parent_grad(self, parent):
        return 1

class Mul(Node):
    def __init__(self, parent1=None, parent2=None) -> None:
        super().__init__(parent1, parent2)
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
