# -*-coding:utf-8-*-

import numpy as np

np.random.seed(1111)


# softmax
def soft_max(logits):
    """
    为了exp不溢出(数值稳定),需要减掉最大值
    输入序列长度与输出序列长度相等
    :param logits:
    :return:
    """
    max_value = np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits - max_value)
    exp_sum = np.sum(exp, axis=1, keepdims=True)
    result = exp / exp_sum
    return result


def toy_nw(x_, w_):
    u_ = np.matmul(x_, w_)
    y_ = soft_max(u_)
    return y_


# CTC prob
def forward(y_, labels_):
    """
    采用动态规划求解p(l|X) X:[time_step,any_dim]
    :param y_: e.g.rnn(X) [time_step,num_class+1]
    :param labels_: l [2*len(l)+1,]
    :return: alpha 概率矩阵 [time_step,2*len(l)+1]
            p(l|X) = alpha[-1,-1]+alpha[-1,-2]
    """
    T, V = y_.shape
    L = len(labels_)
    alpha = np.zeros([T, L])

    # init
    alpha[0, 0] = y_[0, labels_[0]]
    alpha[0, 1] = y_[0, labels_[1]]

    for t in range(1, T):
        for i in range(L):
            s = labels_[i]

            a = alpha[t - 1, i]
            if i - 1 >= 0:
                a += alpha[t - 1, i - 1]
            if i - 2 >= 0 and s != 0 and s != labels_[i - 2]:
                a += alpha[t - 1, i - 2]

            alpha[t, i] = a * y_[t, s]
    return alpha


def backward(y_, labels_):
    T, V = y_.shape
    L = len(labels_)
    beta = np.zeros([T, L])

    # init
    beta[-1, -1] = y_[-1, labels_[-1]]
    beta[-1, -2] = y_[-1, labels_[-2]]

    for t in range(T - 2, -1, -1):
        for i in range(L):
            s = labels_[i]

            a = beta[t + 1, i]
            if i + 1 < L:
                a += beta[t + 1, i + 1]
            if i + 2 < L and s != 0 and s != labels_[i + 2]:
                a += beta[t + 1, i + 2]

            beta[t, i] = a * y_[t, s]

    return beta


def gradient(y_, labels_):
    """
    论文(15)式
    :param y_: [time_step,num_class+1]
    :param labels_: [2*len(l)+1,]
    :return:
    """
    T, V = y.shape
    L = len(labels_)

    alpha_ = forward(y_, labels_)
    beta_ = backward(y_, labels_)
    p = alpha_[-1, -1] + alpha_[-1, -2]

    grad_ = np.zeros([T, V])
    for t in range(T):
        for s in range(V):
            lab = [i for i, c in enumerate(labels_) if c == s]
            for i in lab:
                grad_[t, s] += alpha_[t, i] * beta_[t, i]
            grad_[t, s] /= (y[t, s] ** 2)

    grad_ /= p
    return grad_


def check_gradient(y_, labels_, toleration):
    """
    grad = (f(x+delta)-f(x-delta)) / (2*delta)
    :param y_:
    :param labels_:
    :param toleration:
    :return:
    """
    grad_1 = gradient(y_, labels_)
    delta_ = 1e-10

    for w_ in range(y.shape[0]):
        for v_ in range(y.shape[1]):

            original = y[w_, v_]

            y[w_, v_] = original + delta_
            alpha_ = forward(y_, labels_)
            log_p1 = np.log(alpha_[-1, -1] + alpha_[-1, -2])

            y[w_, v_] = original - delta_
            alpha_ = forward(y_, labels_)
            log_p2 = np.log(alpha_[-1, -1] + alpha_[-1, -2])

            y[w_, v_] = original
            grad_2 = (log_p1 - log_p2) / (2 * delta_)
            # print(grad_2)
            if np.abs(grad_1[w_, v_] - grad_2) > toleration:
                print('[%d, %d]：%.2e' % (w_, v_, np.abs(grad_1[w_, v_] - grad_2)))


def gradient_logits_native(y_, labels_):
    """
    gradient by back propagation
    :param y_:
    :param labels:
    :return: 对y sotfmax 之前的u求梯度,u参见 code line:24
    """
    y_grad_ = gradient(y_, labels_)

    sum_y_grad_ = np.sum(y_grad_ * y, axis=1, keepdims=True)
    u_grad_ = y_ * (y_grad_ - sum_y_grad_)

    return u_grad_


def gradient_logits(y_, labels_):
    """
    将 func gradient() 与 func gradient_logits_native()合并
    :param y_:
    :param labels_:
    :return:
    """
    T_, V_ = y_.shape

    alpha_ = forward(y_, labels_)
    beta_ = backward(y_, labels_)
    p_ = alpha_[-1, -1] + alpha_[-1, -2]

    u_grad_ = np.zeros([T_, V_])
    for t in range(T_):
        for s in range(V_):
            lab = [i for i, c in enumerate(labels_) if s == c]
            for i in lab:
                u_grad_[t, s] += alpha_[t, i] * beta_[t, i]
            u_grad_[t, s] /= (y_[t, s] * p_)
    u_grad_ -= y_
    return u_grad_


def check_gradient_logits(u_, labels_, toleration=1e-3):
    """
    grad = (f(x+delta)-f(x-delta)) / (2*delta)
    :param u_: sotfmax 之前的序列值
    :param labels_:
    :param toleration: 误差阈值
    :return:
    """
    grad_1 = gradient_logits(soft_max(u_), labels_)

    delta_ = 1e-10
    for w in range(u_.shape[0]):
        for v in range(u_.shape[1]):
            original = u_[w, v]

            u_[w, v] = original + delta_
            y_ = soft_max(u_)
            alpha_ = forward(y_, labels_)
            log_p1 = np.log(alpha_[-1, -1] + alpha_[-1, -2])

            u_[w, v] = original - delta_
            y_ = soft_max(u_)
            alpha_ = forward(y_, labels_)
            log_p2 = np.log(alpha_[-1, -1] + alpha_[-1, -2])

            u_[w, v] = original
            grad_2 = (log_p1 - log_p2) / (2 * delta_)
            if np.abs(grad_1[w, v] - grad_2) > toleration:
                print('[%d, %d]：%.2e, %.2e, %.2e' % (w, v, grad_1[w, v], grad_2, np.abs(grad_1[w, v] - grad_2)))


if __name__ == '__main__':
    T, V = 12, 5
    m, n = 6, V
    # 输入序列
    x = np.random.random([T, m])

    # 样例模型参数
    w = np.random.random([m, n])
    # 模型输出(特征抽取)
    y = toy_nw(x, w)
    # print(y)
    # print(np.sum(y, axis=1, keepdims=True))

    # l
    labels = [0, 3, 0, 3, 0, 4, 0]
    alpha = forward(y, labels)
    # p(l|X)
    f_p = alpha[-1, -1] + alpha[-1, -2]
    print('forward prob:', f_p)
    beta = backward(y, labels)
    b_p = beta[0, 0] + beta[0, 1]
    print('backward prob', b_p)

    # grad
    grad = gradient(y, labels)
    # 误差没有大于1e-5的
    check_gradient(y, labels, 1e-5)

    # 对sotfmax之前的u求梯度
    log_grad1 = gradient_logits_native(y, labels)
    print(log_grad1)
    print('\n')
    log_grad2 = gradient_logits(y, labels)
    print(log_grad2)

    check_gradient_logits(np.matmul(x,w), labels, 1e-6)
