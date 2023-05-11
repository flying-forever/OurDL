import sys
sys.path.append('../../..')
from ourdl.core import Varrible, default_graph
from ourdl.ops import Add, Mul, ValueLoss
from ourdl.ops import LeakyRelu as Relu

def build_graph_2():
    '''一层全连接层，使用relu激活层，但是慢很多，损失可以降到3~4'''
    # 1 全连接层
    inputs = []
    hide = []
    outs = []
    # 1.1 输入节点
    for i in range(13):
        x = Varrible()
        inputs.append(x)
    # 1.2 一层隐藏节点
    for i in range(10):
        muls = []
        for j in range(len(inputs)):
            w = Varrible(train=True)
            muls.append(Mul(w, inputs[j]))
        b = Varrible(train=True)
        hide.append(Relu(Add(parents=muls+[b])))
        # hide.append(Add(parents=muls+[b]))
    # 1.3 输出节点
    for i in range(1):
        muls = []
        for j in range(len(hide)):
            w = Varrible(train=True)
            muls.append(Mul(w, hide[j]))
        b = Varrible(train=True)
        outs.append(Add(parents=muls+[b]))
    # 2 标签和损失
    label = Varrible()
    loss = ValueLoss(parents=outs+[label])
    # 3 特殊节点
    default_graph.set_nodes(nodes_in=inputs, nodes_out=outs, node_loss=loss, node_label=label)

    return default_graph

def build_graph():
    '''单纯多对一映射，平均损失在7~10附近'''
    # 1 搭建计算图
    x = []
    mul = []
    for i in range(13):
        x_ = Varrible()
        w_ = Varrible(train=True)
        mul_ = Mul([x_, w_])
        x.append(x_)
        mul.append(mul_)
    b = Varrible(train=True)
    add = Add(parents=mul+[b])
    label = Varrible()
    loss = ValueLoss([add, label])
    # 2 设置计算图的特殊节点
    default_graph.set_nodes(nodes_in=x, nodes_out=[add], node_loss=loss, node_label=label)

    return default_graph
