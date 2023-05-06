'''
@ problems
1 时而成功，时而失败，什么原因？
  - 参数的初始化？
  - 训练数据的随机性？
2 似乎relu折线的上下移动容易，而左右移动少。
@ 提示
1 开启训练中show_ax函数的调用，可以显示拟合过程动画。
'''
import sys
sys.path.append('../..')
from ourdl.core import Varrible
from ourdl.ops import Mul, Add, Relu
from ourdl.ops.loss import ValueLoss

import matplotlib.pyplot as plt
from matplotlib.pylab import scatter
import numpy as np

def show_ax(ax):
    # 1 画真实的二次函数曲线
    show_x = np.linspace(-1, 2, 60, endpoint=True)
    show_y = [x_one * x_one for x_one in show_x]
    ax.plot(show_x, show_y)
    # 2 画模型拟合的曲线
    show_x = np.linspace(-1, 2, 6, endpoint=True)
    show_y = []
    for x_one in show_x:
        x.set_value(x_one)
        add_21.clear()
        add_21.forward()
        show_y.append(add_21.value)
    ax.plot(show_x, show_y)
    plt.pause(0.01)
    ax.cla()

if __name__ == '__main__':
    # 1 搭建计算图
    # 1.1 线性变换一
    x = Varrible()
    w_11 = Varrible()
    w_12 = Varrible()
    mul_11 = Mul([x, w_11])
    mul_12 = Mul([x, w_12])
    b_11 = Varrible()
    b_12 = Varrible()
    add_11 = Add([mul_11, b_11])
    add_12 = Add([mul_12, b_12])
    # 1.2 激活函数 --> 非线性变换
    relu_11 = Relu([add_11])
    relu_12 = Relu([add_12])
    # 1.3 线性变换二
    w_21 = Varrible()
    w_22 = Varrible()
    mul_21 = Mul([relu_11, w_21])
    mul_22 = Mul([relu_12, w_22])
    b_21 = Varrible()
    add_21 = Add([mul_21, mul_22, b_21])
    # 1.4 损失函数
    label = Varrible()
    loss = ValueLoss([label, add_21])

    # 2 参数初始化
    import random
    params = [w_11, w_12, b_11, b_12, w_21, w_22, b_21]
    for param in params:
        param.set_value(random.uniform(-1, 1))
    # 2.1 手工赋值拟合
    # values = [1, 2, 0, -2, 1, 1, 0]
    # for i in range(len(params)):
    #     params[i].set_value(values[i])
    print([param.value for param in params])

    # 3 生成数据
    data_x = [random.uniform(0, 2) for i in range(3000)]  # 似乎实数比离散的[0, 1, 2]要好
    data_label = [x * x for x in data_x]

    # 4 开始训练
    fig = plt.figure()
    ax = fig.subplots()
    losses = []
    for i in range(len(data_x)):
        x.set_value(data_x[i])
        label.set_value(data_label[i])
        loss.forward()
        for param in params:
            param.get_grad()
            param.update(lr=0.01)
        print(f'[{i}]:loss={loss.value},', [param.value for param in params])
        losses.append(loss.value)
        loss.clear()
        if i % 50 == 0:
            # show_ax(ax)  # 开启拟合过程动画
            pass
    
    # 5 画出模型函数图像
    import matplotlib.pyplot as plt
    from matplotlib.pylab import scatter
    import numpy as np
    # 5.1 画真实的二次函数曲线
    show_x = np.linspace(-1, 2, 60, endpoint=True)
    show_y = [x_one * x_one for x_one in show_x]
    plt.plot(show_x, show_y)
    # 5.2 画模型拟合的曲线
    show_x = np.linspace(-1, 2, 6, endpoint=True)
    show_y = []
    for x_one in show_x:
        x.set_value(x_one)
        add_21.clear()
        add_21.forward()
        show_y.append(add_21.value)
    scatter(show_x, show_y)
    plt.plot(show_x, show_y)
    plt.show()

    # 6 画出训练过程中loss的变化曲线
    show_x = [i for i in range(len(losses))]
    show_y = [_ for _ in losses]
    plt.plot(show_x, show_y)
    plt.show()
        