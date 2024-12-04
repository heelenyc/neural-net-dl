# Welcome to neural network!

本文是学习笔记，[原文](http://neuralnetworksanddeeplearning.com/index.html)

### 全连接层架构的神经网络

#### 反向传播算法

反向传播的基本原理是导数求极值的原理（在导数的方向上变动变量，可以让结果更小），让调参的方向跟Cost变小一致；
在一个特定的N(x，y)样本中，代价函数是参数的一个函数，二次代价函数是其中一个：

$$
C(w,b) = \frac{1}{2n}\sum||a - y|| \qquad  其中 a = w*x + b，求和对象是误差的模
$$

根据链式求导，定义$f_{L}$深度为L层的网络最后一层的梯度因子，它是从代价函数$C(a_{L})$传递到L层带权输出$z_L$的偏导，结合激活函数$a_{L} = \sigma({z_{L}})$ :

$$
f_{L} = \frac{\partial C}{\partial z_{L}} = (a_{L}-y)*\sigma^,(z_{L})\qquad  (1)
$$

因为 $z_L = w_L * x + b_L$，所以：

$$
\frac{\partial C}{\partial b_{L}} = f_L * 1  \qquad
\frac{\partial C}{\partial w_{L}} = f_L @ (a_{L-1})^T
$$

继续向L-1层传播

$$
f_{L-1} = \frac{\partial C}{\partial z_{L-1}} = (w_{L})^T @ f_{L} * \sigma^,(z_{L-1})
$$

$$
\frac{\partial C}{\partial b_{L-1}} = f_{L-1} * 1  \qquad
\frac{\partial C}{\partial w_{L-1}} = f_{L-1} @ (a_{L-2})^T
$$

#### 实验结果说明

* 代码
  
  1. v1 版本，面向对象的版本
  2. v2 版本，集中在network里计算，没有按照预期那样出现更好的性能表现
* 关于激活函数
  
  1. 激活函数以及对应的导数值域在0到1之间分布，避免反向传播计算梯度时出现爆炸
  2. 在计算忽略最后一层的sigmoid激活函数在梯度中的偏导，意外获得了更快的学习，但理论上是有问题的，可能带来更大的震荡；
  3. 激活函数以及对应的导数要在0到1之间分布，其一用于归一，其二可以避免爆炸
  4. 激活函数要求连续可导，为了拟合计算（求导）可行
  5. 对于sigmoid函数，当z的绝对值大于4之后，很迅速的贴近y=0和y=1两条线，在那个范围切线变化的很慢，所以带权输出z过大的神经元学习很慢
  6. sigmoid激活函数的导函数阈值是(0,0.25]，所以随着深度加大，反向传播中浅层的梯度迅速消失，所有有*tanh*、*relu*以及*sinh*等替代函数
     ![Alt](./assets/sigmoid.png)
* 关于层数和神经元数量
  
  1. 更深的隐藏层或者更多的隐藏神经元不一定带来更好的效果
* 关于随机样本迭代
  
  1. 小样本随机梯度可以加快学习，因为随机小样本迭代的特点：快速获得特征，小步调整，更敏捷；
  2. 小样本的大小取值并不是越小越好，可能不能及时获取到特征
  3. 小样本太大效果也不一定好，可能会拟合更多的非有效特征导致**过拟合**问题（跟样本拟合很好，代价很小，但是验证精度反而更低）
* 针对识别数字这个问题，全连接层架构的准确率极限在97%左右
* 效果跟初始化的参数有很大关系，开始效果比较差的情况下，学习进度没谱，可能要迭代很多次才能有明显进展，可能跟上述激活函数第六条特性有关；

#### 其他

* exp(z) 容易溢出，没有特别好的解决办法；更换激活函数；

### 改进神经网络的技术

#### 交叉熵代价函数

**前期与预期差距大时，缓解学习缓慢的问题**

之前提到，如果带权输出$z_{L}$的值比较大，sigmoid的导函数值将很小，会导致学习缓慢；

从反向传播的公式1看，在导函数$\sigma^,(z_{L})$前还有个代价函数对${a_{L}}$的偏导部分；能不能抵消掉？

所以**交叉商代价函数**被提出来，其中要的特性就是在计算$f_{L}$时正好把$\sigma^,(z_{L})$约掉了，而且剩下一个与误差正相关的梯度因子（误差越大，梯度越大），以此达到优化梯度学习的目的

$$
C = -\frac{1}{n}\sum\{[y*lna + (1-y)*ln(1-a)]-[y*lny+(1-y)ln(1-y)]\} = -\frac{1}{n}\sum[y*ln\frac{a}{y} + (1-y)*ln\frac{1-a}{1-y}]
$$

根据对数函数的图可以看出，y和1-y都在（0，1）之间，当a越靠近y时，可以认为围出来的面积（负的）越最大，所以代价也就越小，但是对数部分有个最小值，并不能等于零，所以配个常数部分。
具体操作求导时，常数部分可以忽略。这个代价函数对输出a求导的结果正好是

$$
\frac{(a-y)}{a*(1-a)} = \frac{(a-y)}{\sigma^{,}(z)}
$$

所以，在这个代价函数的支持下，$f_{L} = \frac{\partial C}{\partial z_{L}} = (a_{L}-y)$，加快了速度，也简化了计算。
得到的启示是，理论上选择合适的代价函数，可以简化反向传播的计算。

**交叉熵改进的是学习的速率，特别是学习输出与预期相差较大时，天然的加快学习；对学习的精度有改进**

问题来了：什么时候选择二次代价函数，什么时候选择交叉熵代价函数

**应该是取决于输出层的激活函数，如果输出层激活函数是S型的，交叉熵更合适，如果线性的，二次代价也行，其实就是看实际需不需要激活函数的偏导部分**

#### 柔性最大值

需要将输出层解释为概率分布时，可以使用**柔性最大值**作为激活函数，并且结合对数似然代价函数:

$$
a_{j} = \frac{e^{z_{j}}}{\sum\limits_{k} e^{z_{k}}}  \qquad  C = -lna_{y}^{L}
$$

aj是结果是为j的概率。显然该层所有输出节点a之和是1。这对组合跟交叉熵一样能解决学习缓慢的问题。

#### 规范化缓解过度拟合问题

模型在训练样本和测试样本上表现的差异可以作为拟合的度量，如果在测试样本上精度表现明显降低，可能就发生了**过度拟合**（模拟高度匹配训练样本，但是并不是全部聚焦在关键特征上，结果泛化能力不够）

**降低过度拟合最直接的办法就是增加有效的训练规模**，但通常这个是有成本的。

另一种途径是降低网络规模，避免学习到训练集里与测试里不相关的无效特征，这个问题应该是没法量化，可操作性不太强；而且舍弃了大模型更强的表达能力。

#### 权重衰减（L2规范化）

在原始的代价函数里加入一个规范化项，跟所有权重有关系：

$$
C = C_{0} + \frac{\lambda}{2n}\sum\limits_{w}{w^{2}}
$$

只管上理解，增加这个规范化项之后，显然在梯度下降是w会下降的更快，跟名称符合

$$
w = w-w^{,}_{0}-\frac{\lambda\eta}{n}w \qquad  w^{,}_{0} 是原来的梯度迭代值
$$

这个为什么可以控制过拟合，原理貌似不清楚。。。。但实际就是可以，被人诟病。。。

规范化的目的**避免过度拟合，提高分类准确率**，另外还能提高网络的训练稳定性，即多次训练的效果差别不会太大。

#### 其他规范化技术

**L1规范化** 另外一种规范项的添加方式

**弃权** 临时删除部分隐藏节点，类似训练多个网络然后形成组合判断

**人为增加训练样本** 微调训练样本作为新的样本，减少噪音加强特征；

#### 参数初始化（归一化）

**解决初期可能的进展缓慢**

默认的标准正太分布为什么不是个好的分布？ 因为会导致带权输出z太大，而激活输出就基本接近1或者0，变得饱和，饱和之后梯度带来的变化很小，学习进展缓慢。

从直观上理解，开始随机化参数时，权重部分w其实是不需要太大差异，差异太大导致需要耗费无畏的学习，并且可能带来梯度爆炸。

根据正太分布的可加性，为了避免多个参数正太叠加之后阈值变宽，在初始化参数时利用输入个数预先收窄参数的正太分布。即用W ～ N（0，1/len(w)）来初始化权重参数。

### 其他技术

#### 其他随机梯度下降算法

* hessian技术，比仅仅考量一阶导数，还考虑二阶导数，即梯度本身的变化情况
* 基于momentum的梯度下降，增加摩擦系数
* tanh 双曲正切激活函数
* ReLU 激活函数，$a = max(0,w)$ 当带权输出为负数时，神经元在这种情况下不起作用，也不参与学习，感觉非常激进。

### 神经网络可以拟合任何函数

**Functions describe all the world, and neural networks describe all the funtions!**
-_-|||......

### 深度网络很难学习

**因为链式求导的原因，深度网络的梯度里有多个激活函数的偏导，而这个一般都是小于1的**

### 深度学习

#### 卷积神经网络

区别之前的做法是保留二维的特征，通过局部感受的方式映射。

