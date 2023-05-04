if __name__ == '__main__':
    import sys
    sys.path.append('../..')
    import matplotlib.pyplot as plt
    from matplotlib.pylab import scatter
    import numpy as np
    from ourdl.ops import Relu

    x = np.linspace(-5, 5, 20, endpoint=True)
    y = map(lambda x : Relu.relu(x), x)
    y = np.array(list(y))
    plt.plot(x, y)  # 画函数线
    scatter(x, y)  # 标出数据点
    plt.show()
