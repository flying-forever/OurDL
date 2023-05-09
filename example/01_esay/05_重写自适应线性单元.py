'''
1. 感觉逻辑简介清晰了很多。
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
    default_graph.set_nodes(nodes_in=[x], nodes_out=[add], node_loss=loss, node_label=label)
    
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
        # 前向传播 --> 计算各节点的值
        loss = graph.forward([data_x[i]], data_label[i])
        # 反向传播 --> 更新梯度
        graph.backward()
        print(f'loss:{loss}', ["{:.2f}".format(node.value) for node in graph.nodes])
        # 清除梯度信息和运算节点的值 --> 为下次计算准备
        graph.clear()
    # 4 推理
    results = []
    for x in range(5):
        results.append(graph.predict([x]))
    print(f'results:{results}')
