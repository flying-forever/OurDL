from ..core import Node

class Add(Node):
    def __init__(self, parent1=None, parent2=None) -> None:
        super().__init__(parent1, parent2)
    def compute(self):
        self.value = self.parent1.value + self.parent2.value

class Mul(Node):
    def __init__(self, parent1=None, parent2=None) -> None:
        super().__init__(parent1, parent2)
    def compute(self):
        self.value = self.parent1.value * self.parent2.value
