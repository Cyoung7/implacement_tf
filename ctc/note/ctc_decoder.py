# -*-coding:utf-8-*-
import numpy as np
from collections import defaultdict
from ctc.note.ctc_base import soft_max

np.random.seed(1111)
n_inf = - np.float('inf')


def _logsumexp(a, b):
    """
     np.log(np.exp(a) + np.exp(b))
    :param a:
    :param b:
    :return:
    """

    if a < b:
        a, b = b, a

    if b == n_inf:
        return a
    else:
        return a + np.log(1 + np.exp(b - a))


def logsumexp(*args):
    """
    from scipy.special import logsumexp
    :param args:
    :return:
    """
    res = args[0]
    for e in args[1:]:
        res = _logsumexp(res, e)
    return res


def remove_blank(labels, blank=0):
    new_labels = []
    # remove repeat
    previous_c = None
    for c in labels:
        if c != previous_c:
            new_labels.append(c)
            previous_c = c
    # remove blank
    new_labels = [c for c in new_labels if c != blank]
    return new_labels


def insert_blank(labels, blank=0):
    new_labels = [blank]
    for i in labels:
        new_labels.extend([i, blank])
    return new_labels


def greedy_decode(y_, blank=0):
    """
    贪婪搜索，每一步找出概率最大的那个类别
    :param y_: [time_step,num_class+1]
    :param blank:
    :return: 解码序列，解码去重序列
    """
    raw_result = np.argmax(y_, axis=1)
    result = remove_blank(raw_result, blank)
    return raw_result, result


def beam_decode(y_, beam_size_=10, blank=0):
    """
    根据序列得分保留beam_size个高得分序列
    :param y_: [time_step,num_class+1]
    :param beam_size_: 每一个step取保留decode序列数的上限
    :return: top beam_size 序列，及其得分
    """
    T_, V_ = y_.shape
    # 作为得分,概率越大得分越高
    log_y_ = np.log(y_)

    beam_ = [([], 0)]
    # every time_step
    for t in range(T_):
        new_beam_ = []
        # 将t-1 time_step的所有保留序列,分别添加当前step的每个字符
        # 更新序列得分
        for prefix, score_ in beam_:
            for i in range(V_):
                new_prefix_ = prefix + [i]
                new_score = score_ + log_y_[t, i]
                new_beam_.append((new_prefix_, new_score))

        # 保留beam_size个得分高的序列
        new_beam_.sort(key=lambda x: x[1], reverse=True)
        beam_ = new_beam_[:beam_size_]

    new_beam_ = []
    for string_, score_ in beam_:
        new_beam_.append((remove_blank(string_, blank), score_))

    return new_beam_


def prefix_beam_decode(y_, beam_size_=10, blank=0):
    """

    :param y_:
    :param beam_size_:
    :param blank:
    :return:
    """
    T_, V_ = y_.shape
    log_y_ = np.log(y_)

    beam_ = [(tuple(), (0, n_inf))]
    for t in range(T_):
        new_beam_ = defaultdict(lambda: (n_inf, n_inf))

        for prefix, (p_b, p_nb) in beam_:
            for i in range(V_):
                p_ = log_y_[t, i]

                if i == blank:
                    new_p_b_, new_p_nb_ = new_beam_[prefix]
                    new_p_b_ = logsumexp(new_p_b_, p_b + p_, p_nb + p_)
                    new_beam_[prefix] = (new_p_b_, new_p_nb_)
                    continue
                else:
                    end_t_ = prefix[-1] if prefix else None

                    new_prefix = prefix + (i,)
                    new_p_b_, new_p_nb_ = new_beam_[new_prefix]
                    if i != end_t_:
                        new_p_nb_ = logsumexp(new_p_nb_, p_b + p_, p_nb + p_)
                    else:
                        new_p_nb_ = logsumexp(new_p_b_, p_b + p_)
                    new_beam_[new_prefix] = (new_p_b_, new_p_nb_)

                    if i == end_t_:
                        new_p_b_, new_p_nb_ = new_beam_[prefix]
                        new_p_nb_ = logsumexp(new_p_nb_, p_nb + p_)
                        new_beam_[prefix] = (new_p_b_, new_p_nb_)
        beam_ = sorted(new_beam_.items(), key=lambda x: logsumexp(*x[1]), reverse=True)
        beam_ = beam_[:beam_size_]

    new_beam_ = []
    for string_, score_ in beam_:
        new_beam_.append((remove_blank(string_, blank), score_))

    return new_beam_


if __name__ == '__main__':
    y = soft_max(np.random.random([20, 6]))
    print(y)
    rr, rs = greedy_decode(y)
    print(rr)
    print(rs)
    # 束搜索
    beam = beam_decode(y, beam_size_=10)
    for per in beam:
        print(per)
    # 前缀束搜索
    beam = prefix_beam_decode(y, beam_size_=10)
    for per in beam:
        print(per)
