import sys
sys.path.append('../..')
import ourdl as od

if __name__ == '__main__':
    # 搭建计算图: y=2x+1
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
    print(f"y: {y}")
    # 反向传播求梯度
    w_grad = w1.get_grad()
    x_grad = x1.get_grad()
    print(f"w_grad: {w_grad}, x_grad: {x_grad}")

    '''
    请输入x：3
    y: 7
    w_grad: 3, x_grad: 2
    '''
