import sys
sys.path.append('../..')
import ourdl as od

if __name__ == '__main__':
    # 搭建计算图
    x1 = od.core.Varrible()
    w1 = od.core.Varrible()
    mul = od.ops.Mul(x1, w1)
    b = od.core.Varrible()
    add = od.ops.Add(mul, b)
    # 给参数赋值
    w1.set_value(2)
    b.set_value(1)
    # 使用计算图计算
    x1.set_value(int(input("请输入x：")))
    y = add.forward()
    print(y)

    '''
    请输入x：2
    5
    '''
