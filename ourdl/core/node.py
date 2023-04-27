class Node:
    def __init__(self, parent1=None, parent2=None) -> None:
        self.parent1 = parent1
        self.parent2 = parent2
        self.value = None
    def set_value(self, value):
        self.value = value
    def compute(self):
        '''抽象方法，在具体的类中重写'''
        pass
    def forward(self):
        for parent in [self.parent1, self.parent2]:
            if parent.value is None:
                parent.forward()
        self.compute()
        return self.value

class Varrible(Node):
    def __init__(self) -> None:
        super().__init__()