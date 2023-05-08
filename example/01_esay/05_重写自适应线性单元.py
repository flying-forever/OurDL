'''
1. 与第一次的代码量差不多，但是成功把计算图的搭建从主函数中抽离出来。
2. 直接访问类graph的nodes等属性并不方便，因为不知道node的类型，没有代码补全。
   这就是将对属性的操作都封装成一个方法的好处吗？
3. 目前实现中，手动"设置计算图的特殊节点"的方式并不方便。
'''
import sys
sys.path.append('../..')
from ourdl.core import Varrible
from ourdl.ops import Mul, Add
from ourdl.ops.loss import ValueLoss
from ourdl import default_graph
import random

def build_graph():
    # 1 搭建计算图
    x = Varrible()
    w = Varrible(train=True)
    b = Varrible(train=True)
    add = Add(Mul([x, w]), b)
    label = Varrible()
    loss = ValueLoss(label, add)
    # 2 设置计算图的特殊节点
    default_graph.nodes_in = [x]
    default_graph.nodes_out = [add]
    default_graph.node_loss = loss
    default_graph.node_label = label

    return default_graph

if __name__ == '__main__':
    # 1 创建计算图对象
    graph = build_graph()
    graph.init_params()
    # 2 生成训练数据
    data_x = [random.uniform(-10, 10) for i in range(200)]  # 按均匀分布生成[-10, 10]范围内的随机实数
    data_label = [2 * data_x_one + 1 for data_x_one in data_x]
    # 3 训练
    for i in range(len(data_x)):
        # 3.1 为输入和标签赋值
        graph.nodes_in[0].set_value(data_x[i])
        graph.node_label.set_value(data_label[i])
        # 3.2 前向传播 + 反向传播更新参数
        loss = graph.node_loss.forward()
        graph.backward()
        print(f'loss:{loss}', ["{:.2f}".format(node.value) for node in graph.nodes])
        graph.node_loss.clear()
