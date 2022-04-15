# -*- coding: utf8 -*-
#

# 求8的立方根

def newton(number):
    cur_num = 1e-2

    while True:
        if abs(cur_num * cur_num * cur_num - number) > 0.01:
            cur_num += 1e-2
            print(cur_num)
        else:
            break
    print('final: ', cur_num)


if __name__ == '__main__':
    newton(12)
