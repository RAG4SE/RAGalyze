import trace
import traceback
import sys

def level1():
    level2()

def level2():
    level3()

def level3():
    try:
        raise ValueError('测试错误')
    except ValueError as e:
        print(traceback.format_exc())
        # raise Exception(traceback.format_exc())

level1()
# try:
# except Exception as e:
#     print('=== 完整调用链测试 ===')
#     print(traceback.format_exc())

a = {
    1: 2,
    2: 3    
}

del a[1]
print(a)