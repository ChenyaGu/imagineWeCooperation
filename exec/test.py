import functools as ft
a = []
for i in [1, 2, 4]:
    funadd = lambda x, y: x + y
    fun = ft.partial(funadd,y=i)
    a.append(fun)
print(a[0](0))
print(a[1](0))
print(a[2](0))
