import sys
sys.path.append('../..')
from ourdl.core import Varrible, default_graph
from ourdl.ops import Add, Mul, ValueLoss
from ourdl.ops import LeakyRelu as Relu
import random

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

def build_graph_2():
    '''一层全连接层，精度与多对一映射相似，但是慢很多'''
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

def get_data():
    '''
    读取波士顿房价数据集，返回输入data_in和标签data_label
    @ return
    1. data_in --> list(: float)
    2. data_label --> float
    '''
    data_in = []
    data_label = []
    print('start reading...')
    with open('E://dataset//波士顿房价//housing.data', 'r') as f:
        # 打乱数据集
        lines = f.readlines()
        random.shuffle(lines)
        for line in lines:
            line = line.split(' ')
            line = [float(d.strip('\n')) for d in line if d != '']  # 连续的空格会产生空串'';最后一个串尾会有'\n'
            data_in.append(line[:-1])
            data_label.append(line[-1])
    assert len(data_in) == len(data_label), "输入与标签的样本数不同！"
    print(f'read success, len={len(data_label)}')
    return data_in, data_label

if __name__ == '__main__':

    # 1 搭建计算图与参数初始化
    graph = build_graph_2()
    graph.init_params()
    print(f'size of graph:{len(graph.nodes)}')

    # 2 读取训练数据
    data_in, data_label = get_data()
    print(data_in[0], data_label[0])
    # 2.1 数据集的离散程度
    avg = sum(data_label) / len(data_label)
    losses = [label - avg for label in data_label]
    losses = [loss if loss >= 0 else -loss for loss in losses]
    avg_loss = sum(losses) / len(losses)
    print(f'avg {avg}, avg_loss {avg_loss}')

    # 3 训练
    # 3.1 训练前的损失
    losses = []
    predicts = []
    for i, x in enumerate(data_in):
        graph.clear()
        loss = graph.forward(x=x, label=data_label[i])
        predict = graph.predict(x=x)
        losses.append(loss)
        predicts.append(predict)
    print('-------------------------------------------------')
    # 3.2 训练前的拟合效果
    import matplotlib.pyplot as plt
    x = [i for i in range(len(data_label))]
    n = 100
    plt.plot(x[:n], data_label[:n])
    plt.plot(x[:n], predicts[:n])
    plt.show()
    # 3.2 进行训练
    for epoch in range(50):
        losses_t = []
        for i, x in enumerate(data_in):
            loss = graph.forward(x=x, label=data_label[i])
            graph.backward()
            graph.clear()
            losses_t.append(loss)
        print(f'epoch {epoch}, avg_loss {sum(losses_t)/len(losses_t)}')
    # 3.3 训练后的损失
    losses_2 = []
    predicts = []
    for i, x in enumerate(data_in):
        predict = graph.predict(x=x)
        loss2 = graph.forward(x=x, label=data_label[i])
        graph.clear()
        predicts.append(predict)
        losses_2.append(loss2)
        # print(f'p {predict[0]}, y {data_label[i]}, loss, {loss2}')
    # 3.4 比较
    print(len(losses_t))
    print(f'start:{sum(losses)/len(losses)}, end:{sum(losses_2)/len(losses_2)}')
    # 3.5 画图
    import matplotlib.pyplot as plt
    x = [i for i in range(len(data_label))]
    n = 100
    plt.plot(x[:n], data_label[:n])
    plt.plot(x[:n], predicts[:n])
    plt.show()
