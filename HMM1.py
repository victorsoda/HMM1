#coding:utf-8

import os
import math
import re
import numpy as np
from scipy.special import logsumexp


TRAIN_PATH = "../data/train/"
VALID_PATH = "../data/valid/"
TEST_PATH = "../data/test/"

# speech: s1, s2, ..., s_N
# word: w1, w2, ..., w_M
# X_t: hidden state, speech
# o_t: output state, word

S1 = {}  # key: speech    value: count
S2 = {}  # key: (speech_i, speech_{i+1})    value: count
SW = {}  # key: (speech_i, word_i)    value: count
P = {}   # key: si    value: P(X1=si)
A = {}   # key: (si, sj)    value: P(X_{t+1}sj | si)
B = {}   # key: (si, wk)    value: P(wk | si)
punc = ['\n', '“', '”', '[', ']']
punc_pattern = r"([。！？\s+]/w)"
# method = 'adding-one'
method = 'good-turing'
fb = 'forward'
# fb = 'backward'


def del_punc(word):
    ret = word
    for p in punc:
        ret = ret.replace(p, '')
    return ret


def del_bracket(speech):
    ret = speech.split(']')[0]
    return ret


def put_stuff_into_count_dict(w, d):
    if w in d:
        d[w] += 1
    else:
        d[w] = 1


def count(word, V):
    if word in V.keys():
        count = V[word]
    else:
        count = 0
    return count


def article_to_sentences(article):
    lines = article.split('\n')
    sentences = []
    for line in lines:
        if re.search(punc_pattern, line) is not None:
            s = re.split(punc_pattern, line)
            s.append("")
            s = ["".join(i) for i in zip(s[0::2], s[1::2])][:-1]
            sentences.extend(s)
        elif len(line) > 3:  # 防止空行
            sentences.append(line)
    return sentences


def sentence_to_wordlist(s):
    s = s.split('  ')[:-1]
    while '' in s:
        s.remove('')
    wordlist = [(del_punc(x.split('/')[0]), del_bracket(x.split('/')[1])) for x in s]
    return wordlist


def r_star(word, V, Nr):
    try:
        if word not in V.keys():  # 没有出现过的词
            return Nr[1]
        r = V[word]
        if r > 11 or r+1 not in Nr.keys():
            return r  # 对于稀疏高频词（它的频率r，但不存在频率r+1的词）拟合结果
        else:
            return (r + 1) * Nr[r+1] / Nr[r]  # 否则使用Good Turing公式近似概率
    except KeyError as e:
        print("KeyError!", V[word])



def run(path):
    """
    对path路径下的所有句子，计算句子的概率。
    :param path: VALID_PATH or TEST_PATH
    :return:
    """
    PPS_list = []
    ccc = 0

    # N0_S2 = 0
    # N0_SW = 0
    #
    #
    # for file in os.listdir(path):
    #     with open(path + file, 'r', encoding='gbk') as f:
    #         # print(file)
    #         article = f.read()
    #         sentences = article_to_sentences(article)
    #         for s in sentences:
    #             wordlist = sentence_to_wordlist(s)
    #             n = len(wordlist)



    for file in os.listdir(path):
        with open(path + file, 'r', encoding='gbk') as f:
            # print(file)
            article = f.read()
            sentences = article_to_sentences(article)
            for s in sentences:
                wordlist = sentence_to_wordlist(s)
                n = len(wordlist)
                if n == 0:
                    continue
                prob = 0
                if fb == 'forward':
                    alpha = np.zeros((N1, n+1))
                    for i in range(N1):
                        start = count(speech_list[i], P)
                        # if method == 'adding-one':
                        #     if start == 0:
                        #         start = 1 / len(P)
                        alpha[i][0] = start
                    for t in range(n):
                        for j in range(N1):
                            for i in range(N1):
                                trans = count((speech_list[i], speech_list[j]), A)  # 转移概率a_{ij}
                                if trans == 0:
                                    if method == 'adding-one':
                                        trans = 1 / N1
                                    elif method == 'good-turing':
                                        trans = (r_star((si, sj), S2, S2_Nr) / S2_N) / (r_star(si, S1, S1_Nr) / S1_N)
                                gen = count((speech_list[i], speech_list[j], wordlist[t][0]), B)    # 生成概率b_{ijk}
                                if gen == 0:
                                    if method == 'adding-one':
                                        gen = 1 / N2
                                    elif method == 'good-turing':
                                        gen = (r_star((si, sj, wk), SW, SW_Nr) / SW_N) / (r_star((si, sj), S2, S2_Nr) / S2_N)
                                alpha[j][t+1] += alpha[i][t] * trans * gen
                    prob = np.sum(alpha[:, n])
                elif fb == 'backward':
                    beta = np.zeros((N1, n+1))
                    for i in range(N1):
                        beta[i][n] = 1
                    for t in range(n-1, -1, -1):
                        for i in range(N1):
                            for j in range(N1):
                                trans = count((speech_list[i], speech_list[j]), A)
                                if trans == 0:
                                    trans = 1 / N1
                                gen = count((speech_list[i], speech_list[j], wordlist[t][0]), B)
                                if gen == 0:
                                    gen = 1 / N2
                                beta[i][t] += beta[j][t+1] * trans * gen
                    for i in range(N1):
                        prob += count(speech_list[i], P) * beta[i][0]
                if prob != 0:
                    PPS = math.pow(prob, -1 / n)
                    PPS_list.append(PPS)
                else:
                    print("prob of sentence = 0!")

            ccc += 1
            if ccc % 1 == 0:
                print(ccc)
                print(np.mean(PPS_list))
                # exit(2)

    ans = np.array(PPS_list)
    return ans


def construct_from_train_set():
    """
    构建训练集对应的词库。
    :return:
    """
    for file in os.listdir(TRAIN_PATH):
        with open(TRAIN_PATH + file, 'r', encoding='gbk') as f:
            # print(file)
            article = f.read()
            sentences = article_to_sentences(article)
            for s in sentences:
                wordlist = sentence_to_wordlist(s)
                n = len(wordlist)
                if n == 0:
                    continue
                X1 = wordlist[0][1]
                put_stuff_into_count_dict(X1, P)
                for i in range(n):
                    oi, Xi = wordlist[i]
                    put_stuff_into_count_dict(Xi, S1)
                    if i < n - 1:
                        oj, Xj = wordlist[i + 1]
                        put_stuff_into_count_dict((Xi, Xj), S2)
                        put_stuff_into_count_dict((Xi, Xj, oi), SW)


construct_from_train_set()

N1 = len(S1)    # total of speeches = 41
N2 = len(S2)    # total of speech pair (speech_i, speech_{i+1}) = 1033
N_sw = len(SW)  # total of (speech, word) pair = 56096
speech_list = list(S1.keys())
P_Nr = {}   # key: 出现的频次c    value:在P中出现c次的词性的个数
S1_Nr = {}
S2_Nr = {}
SW_Nr = {}
P_N = sum(P.values())     # 所有P中词性(key)出现的次数总和，即sum(P.values())
S1_N = sum(S1.values())
S2_N = sum(S2.values())
SW_N = sum(SW.values())


if method == 'adding-one':
    # Pi, 初始概率
    P_total = sum(P.values())
    for key in P.keys():
        P[key] = (P[key] + 1) / (P_total + len(P))
    # A, 转移概率矩阵
    for item in S2.items():
        si, sj = item[0]
        cnt = item[1]
        A[(si, sj)] = (cnt + 1) / (S1[si] + N1)
    # B, 生成概率矩阵
    for item in SW.items():
        si, sj, wk = item[0]
        cnt = item[1]
        B[(si, sj, wk)] = (cnt + 1) / (S2[(si, sj)] + N2)
elif method == 'good-turing':
    for c in S1.values():
        put_stuff_into_count_dict(c, S1_Nr)
    for c in P.values():
        put_stuff_into_count_dict(c, P_Nr)
    for c in S2.values():
        put_stuff_into_count_dict(c, S2_Nr)
    for c in SW.values():
        put_stuff_into_count_dict(c, SW_Nr)
    # Pi, 初始概率
    for key in P.keys():    # TODO: 没有在P中出现过的词性，怎么用Good Turing平滑化？
        P[key] = r_star(key, P, P_Nr) / P_N
    # A, 转移概率矩阵
    for item in S2.items():
        si, sj = item[0]
        cnt = item[1]
        A[(si, sj)] = (r_star((si, sj), S2, S2_Nr) / S2_N) / (r_star(si, S1, S1_Nr) / S1_N)
    # B, 生成概率矩阵
    for item in SW.items():
        si, sj, wk = item[0]
        cnt = item[1]
        B[(si, sj, wk)] = (r_star((si, sj, wk), SW, SW_Nr) / SW_N) / (r_star((si, sj), S2, S2_Nr) / S2_N)


print(N1, N2, N_sw, len(P), len(A), len(B))

# PPS_list_valid = run(VALID_PATH)
# print("PPS(valid):", np.mean(PPS_list_valid))
PPS_list_test = run(TEST_PATH)
print("PPS(test):", np.mean(PPS_list_test))
fb = 'forward'
PPS_list_test = run(TEST_PATH)
print("PPS(test):", np.mean(PPS_list_test))
