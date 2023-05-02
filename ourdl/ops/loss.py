from ..core import Node

class LossFunction(Node):
    '''抽象损失函数类'''
    pass

class ValueLoss(LossFunction):
    '''损失函数：作差取绝对值'''
    def compute(self):
        self.value = self.parent1.value - self.parent2.value
        self.flag = self.value > 0
        # print(f"loss_value:{self.value}, loss_flag:{self.flag}")
        if not self.flag:
            self.value = -self.value
    def get_parent_grad(self, parent):
        a = 1 if self.flag else -1
        b = 1 if parent == self.parent1 else -1
        return a * b
