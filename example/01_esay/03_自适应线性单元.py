import sys
sys.path.append('../..')
from ourdl.core import Varrible
from ourdl.ops import Mul, Add
from ourdl.ops.loss import ValueLoss

if __name__ == '__main__':
    # 搭建计算图
    x = Varrible()
    w = Varrible()
    mul = Mul(parent1=x, parent2=w)
    b = Varrible()
    add = Add(parent1=mul, parent2=b)
    label = Varrible()
    loss = ValueLoss(parent1=label, parent2=add)
    # 参数初始化
    w.set_value(0)
    b.set_value(0)
    # 生成训练数据
    import random
    data_x = [random.uniform(-10, 10) for i in range(1000)]
    data_label = [2 * data_x_one + 1 for data_x_one in data_x]
    # 开始训练
    for i in range(len(data_x)):
        x.set_value(data_x[i])
        label.set_value(data_label[i])
        loss.forward()  # 前向传播 --> 求梯度会用到损失函数的值
        w.get_grad()
        b.get_grad()
        # print(f"x:{x.value}, w_g:{w.grad}, b_g:{b.grad}")
        w.update(lr=0.001)
        b.update(lr=0.002)
        loss.clear()
        print(f"w:{w.value}, b:{b.value}")
    print("最终结果：{:.2f}x+{:.2f}".format(w.value, b.value))
    