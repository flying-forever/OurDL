class One:
    def __init__(self):
        self.one = 1

class node:
    def __init__(self) -> None:
        self.x = One()
        self.y = One()
        z = [self.x, self.y]
        for i in range(len(z)):
            z[i].one += 1
        for z_one in z:
            z_one.one += 1

n = node()
print(n.x.one, n.y.one)
    