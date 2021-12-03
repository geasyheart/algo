## 反向传播实现

> mlp.py

反向传播实现

> torch_mlp.py

使用torch简洁实现.


## 思考

torch_mlp是使用linear将input直接输入，即input直接作为特征来输入即：
```
input：1 * 2
weight：2 * 5
out: 1 * 5 * 1
```

### 我这里转换一种思想:
#### 1. 使用embedding(torch_mlp2.py)
使用embedding，证明在预测上此方法不可行，只能预测已知的x和y，超出后无法预测.


#### 2. 将每一个数字转成一个10维的特征，然后用特征去拟合最终结果

1*10的linear(torch_mlp3.py)

### 结论

如果特征是纯数字的话，直接用linear，没有骚操作






