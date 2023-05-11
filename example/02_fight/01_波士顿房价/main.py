from build_graph import build_graph_2, build_graph
from data import get_data

if __name__ == '__main__':

    def label_differ(labels):
        '''数据集的标签的平均值和离散程度（差的绝对值的和）'''
        avg = sum(data_label) / len(data_label)
        losses = [label - avg for label in data_label]
        losses = [loss if loss >= 0 else -loss for loss in losses]
        avg_loss = sum(losses) / len(losses)
        return avg, avg_loss

    # 1 搭建计算图与参数初始化
    graph = build_graph_2()
    graph.init_params()
    print(f'size of graph:{len(graph.nodes)}')

    # 2 读取训练数据
    data_in, data_label = get_data()
    print(data_in[0], data_label[0])
    # 2.1 数据集的离散程度
    avg, avg_loss = label_differ(data_label)
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
    for epoch in range(20):
        losses_t = []
        for i, x in enumerate(data_in):
            loss = graph.forward(x=x, label=data_label[i])
            graph.backward(lr=0.0001)
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
