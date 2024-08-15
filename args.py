def fun(x1, x2, *, x3, x4, x5):
    return x1 + x2 + x3 + x4 + x5


def fun2(*args, **kwargs):
    return fun(*args, **kwargs)


t = (1, 2)
d = {"x3": 3, "x4": 4, "x5": 5}
print(fun2(*t, **d))
