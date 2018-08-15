# CTC note

paper:https://www.cs.toronto.edu/~graves/icml_2006.pdf

blog:https://distill.pub/2017/ctc/

## 1.关于动态规划的一些理解:

状态转移矩阵 $\alpha$ : shape: [time_step,L]

输入序列 $y$ : shape: [time_step,num_class+1]

输出序列 $l$ : 插入空白字符的 label，label长为 $k$， 序列 $l$的长度为 $2*k+1$

 num_class代表label序列所有可能出现的类别数，+1 是增加一个空白字符类

$y[t,c]$:第t步预测为第c个字符的概率

$\alpha [t,v]$ :输入序列 $y[1:t,:]$ 匹配 $l[1:v]$的概率总和。同时在此节点上:输入序列 $y$ 在第 $t$ 步的激活类别正是输出序列 $l$ 第 $v$ 个字符