import sys
sys.path.append('../..')
from ourdl.ops import Step
from ourdl.core import Varrible
if __name__ == '__main__':
    '''发现：目前实现的运算节点，只能在计算图中使用，因为输入是从父节点获得的。所以单独测试会很麻烦，
    需要先创建计算图。'''
    import matplotlib.pyplot as plt
    import numpy as np
    inputs = Varrible()
    outputs = Step(parent1=inputs)
    # x = list(range(-5, 5, 1))  # range: 步长只能是整数
    x = np.linspace(-5, 5, 20, endpoint=True)  # 在[-5,5]区间中等间隔取20个点
    y = []
    for x_one in x:
        inputs.set_value(x_one)
        outputs.forward()
        y.append(outputs.value)
    # 绘制阶跃函数的图像
    plt.plot(x, y)
    from matplotlib.pylab import scatter
    scatter(x, y)
    plt.show()
