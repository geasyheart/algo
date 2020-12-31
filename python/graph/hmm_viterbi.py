# -*- coding: utf8 -*-

# 参考文档: http://www.hankcs.com/nlp/hmm-and-segmentation-tagging-named-entity-recognition.html

"""

稍微用中文讲讲思路，很明显，第一天天晴还是下雨可以算出来：

1. 定义V[时间][今天天气] = 概率，注意今天天气指的是，前几天的天气都确定下来了（概率最大）今天天气是X的概率，这里的概率就是一个累乘的概率了。

2. 因为第一天我的朋友去散步了，所以第一天下雨的概率V[第一天][下雨] = 初始概率[下雨] * 发射概率[下雨][散步] = 0.6 * 0.1 = 0.06，同理可得V[第一天][天晴] = 0.24 。从直觉上来看，因为第一天朋友出门了，她一般喜欢在天晴的时候散步，所以第一天天晴的概率比较大，数字与直觉统一了。

3. 从第二天开始，对于每种天气Y，都有前一天天气是X的概率 * X转移到Y的概率 * Y天气下朋友进行这天这种活动的概率。因为前一天天气X有两种可能，所以Y的概率有两个，选取其中较大一个作为V[第二天][天气Y]的概率，同时将今天的天气加入到结果序列中

4. 比较V[最后一天][下雨]和[最后一天][天晴]的概率，找出较大的哪一个对应的序列，就是最终结果。

"""

# 隐状态
states = ('Rainy', 'Sunny')
# 观察序列
observations = ('walk', 'shop', 'clean')
# 初始概率（隐状态）
start_probability = {'Rainy': 0.6, 'Sunny': 0.4}
# 转移概率（隐状态）
transition_probability = {
    'Rainy': {'Rainy': 0.7, 'Sunny': 0.3},
    'Sunny': {'Rainy': 0.4, 'Sunny': 0.6},
}
# 发射概率 （隐状态表现为显状态的概率）
emission_probability = {
    'Rainy': {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
    'Sunny': {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
}


# 打印路径概率表
def print_dptable(V):
    print("    ")
    for i in range(len(V)): print("%7d" % i)
    print()
    for y in V[0].keys():
        print("%.5s: " % y)
        for t in range(len(V)):
            print("%.7s" % ("%f" % V[t][y]))
        print()


def viterbi(obs, states, start_p, trans_p, emit_p):
    """

    :param obs:观测序列
    :param states:隐状态
    :param start_p:初始概率（隐状态）
    :param trans_p:转移概率（隐状态）
    :param emit_p: 发射概率 （隐状态表现为显状态的概率）
    :return:
    """
    # 路径概率表 V[时间][隐状态] = 概率
    V = [{}]
    # 一个中间变量，代表当前状态是哪个隐状态
    path = {}

    # 初始化初始状态 (t == 0)
    print('Day 1(walk): ')
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]
        print(y, V[0][y], '=', start_p[y], '*', emit_p[y][obs[0]])
    print('_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _')
    print('V: ', V)
    print('__________________________________________')
    # 对 t > 0 跑一遍维特比算法
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}

        print('Day', t + 1, '(' + obs[t] + ')')
        for y in states:
            # 概率 隐状态 =    前状态是y0的概率 * y0转移到y的概率 * y表现为当前状态的概率
            # (prob, state) = max([(V[t - 1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states])
            tmpList = []
            print()
            for y0 in states:
                tmpList.append((V[t - 1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0))

                print('當天' + y + ': ', "%.7s" % (V[t - 1][y0] * trans_p[y0][y] * emit_p[y][obs[t]]),),
                print('=',),
                print(V[t - 1][y0], '(前一天 ' + y0 + ')', '*',),
                print(trans_p[y0][y], '(' + y0 + ' to ' + y + ')', '*',),
                print(emit_p[y][obs[t]], '(' + y, obs[t] + ')'),
            print('當天' + y + '  最大的是: ', max(tmpList)[0])
            (prob, state) = max(tmpList)

            # 记录最大概率
            V[t][y] = prob
            # 记录路径
            newpath[y] = path[state] + [y]

        # 不需要保留旧路径
        path = newpath
        print('__________________________________________')

    print_dptable(V)
    (prob, state) = max([(V[len(obs) - 1][y], y) for y in states])
    return (prob, path[state])


def example():
    return viterbi(observations,
                   states,
                   start_probability,
                   transition_probability,
                   emission_probability)


print(example())