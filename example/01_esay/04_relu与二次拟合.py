'''
@ problems
1 时而成功，时而失败，什么原因？
  - 参数的初始化？  
  - 训练数据的随机性？
: 参数的初始化似乎对训练效果有着决定性的影响。
2 似乎relu折线的上下移动容易，而左右移动少。

@ 提示
1 开启训练中show_ax_mul函数的调用，可以显示拟合过程动画。

@ 想法 | 问题
1 为什么模型图像有时可以有三个转折点？不应该最多两个转折点吗？
2 将画布分成多块，展示计算图的多个阶段。
'''
import sys
sys.path.append('../..')  # 父目录的父目录
from ourdl.core import Varrible
from ourdl.ops import Mul, Add
from ourdl.ops.loss import ValueLoss
from ourdl.ops import LeakyRelu as Relu
import matplotlib.pyplot as plt
import numpy as np
import random

def show_ax_mul(ax):
    '''
    用于绘制多图动画\n
    @ problem\n
    1 在这里进行一次forward()应该是冗余的，因为在计算损失时已经forward()一次了'''
    # 1 画真实的二次函数曲线
    show_x = np.linspace(-1.5, 2.5, 60, endpoint=True)
    show_y = [x_one * x_one for x_one in show_x]
    ax[2].plot(show_x, show_y)
    # 2 画模型拟合的曲线
    show_x = np.linspace(-1.5, 2.5, 20, endpoint=True)
    shows = {'add1':[], 'add2':[], 'mul1':[], 'mul2':[], 'y':[]}
    for x_one in show_x:
        x.set_value(x_one)
        add_21.clear()
        add_21.forward()
        shows['y'].append(add_21.value)
        shows['add1'].append(add_11.value)
        shows['add2'].append(add_12.value)
        shows['mul1'].append(mul_21.value)
        shows['mul2'].append(mul_22.value)
    ax[2].plot(show_x, shows['y'])
    ax[0].plot(show_x, shows['add1'])
    ax[0].plot(show_x, shows['add2'])
    ax[1].plot(show_x, shows['mul1'])
    ax[1].plot(show_x, shows['mul2'])
    y_0 = [0 for _ in show_x]
    ax[2].plot(show_x, y_0)
    ax[0].plot(show_x, y_0)
    ax[1].plot(show_x, y_0)
    plt.pause(0.01)
    for i in range(ax.shape[0]):
        ax[i].cla()

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
    params = [w_11, w_12, b_11, b_12, w_21, w_22, b_21]
    # for param in params:
    #     param.set_value(random.uniform(-1, 1))
    # 2.1 手工赋值拟合
    # values = [1, 2, 0, -2, 1, 1, 0]
    values = [-0.1571950013426796, -0.1070365984042347, 0.3791639008324807, 0.31960284774415215, 0.4263410176300597, 0.5097967360623379, 0.7597168751185974]
    for i in range(len(params)):
        params[i].set_value(values[i])
    print([param.value for param in params])

    # 3 生成数据
    data_x = [random.uniform(0, 2) for i in range(2000)]  # 似乎实数比离散的[0, 1, 2]要好
    data_label = [x * x for x in data_x]

    # 4 开始训练
    # 4.1 多图绘制模式
    fig = plt.figure(figsize=(15,4))
    ax = fig.subplots(1,3,sharex=True,sharey=False)
    # 4.2 训练，同时绘制动画
    losses = []
    for i in range(len(data_x)):
        x.set_value(data_x[i])
        label.set_value(data_label[i])
        loss.forward()
        for param in params:
            param.get_grad()
            param.update(lr=0.01)
        if i % 200 == 0:
            print(f'[{i}]:loss={loss.value},', [param.value for param in params])
        losses.append(loss.value)
        loss.clear()
        if i % 100 == 0:
            show_ax_mul(ax)  # 多图动画

    # 6 画出训练过程中loss的变化曲线
    show_x = [i for i in range(len(losses))]
    show_y = [_ for _ in losses]
    plt.plot(show_x, show_y)
    plt.show()
        