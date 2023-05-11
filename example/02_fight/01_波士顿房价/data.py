import random

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
