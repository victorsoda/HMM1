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

bi_N = 0   # 所有bigram出现的总次数
S1 = {}  # key: speech    value: count
S2 = {}  # key: (speech_i, speech_{i+1})    value: count
SW = {}  # key: (speech_i, word_i)    value: count
P = {}   # key: si    value: P(X1=si)
A = {}   # key: (si, sj)    value: P(X_{t+1}sj | si)
B = {}   # key: (si, wk)    value: P(wk | si)
punc = ['\n', '“', '”', '[', ']']
punc_pattern = r"([。！？，、（）“”；：《》\s+]/w)"
method = 'adding-one'
# method = 'good-turing'


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


def run(path):
    """
    对path路径下的所有句子，计算句子的概率。
    :param path: VALID_PATH or TEST_PATH
    :return:
    """
    PPS_list = []
    ccc = 0
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
                alpha = np.zeros((N1, n+1))
                for i in range(N1):
                    alpha[i][0] = count(speech_list[i], P)
                for t in range(n):
                    for j in range(N1):
                        for i in range(N1):
                            trans = count((speech_list[i], speech_list[j]), A)
                            if trans == 0:
                                trans = 1 / N1
                            gen = count((speech_list[i], wordlist[t][0]), B)
                            if gen == 0:
                                gen = 1 / N1
                            alpha[j][t+1] += alpha[i][t] * trans * gen
                # print(alpha[:, n].shape)
                # exit(2)
                # print(alpha)
                prob = logsumexp(alpha[:, n])
                if prob != 0:
                    PPS = math.pow(prob, -1/n)
                    PPS_list.append(PPS)
            ccc += 1
            if ccc % 50 == 0:
                print(ccc)

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
                    put_stuff_into_count_dict((Xi, oi), SW)
                    if i < n - 1:
                        oj, Xj = wordlist[i + 1]
                        put_stuff_into_count_dict((Xi, Xj), S2)


construct_from_train_set()

N1 = len(S1)    # total of speeches = 41
N2 = len(S2)    # total of speech pair (speech_i, speech_{i+1}) = 1033
N_sw = len(SW)  # total of (speech, word) pair = 56096
speech_list = list(S1.keys())

# Pi, 初始概率
P_total = sum(P.values())
for key in P.keys():
    P[key] = (P[key] + 1) / (P_total + len(P))
# A, 转移概率矩阵
for item in S2.items():
    si, sj = item[0]
    cnt = item[1]
    A[(si, sj)] = (cnt + 1) / (S1[si] + N1)   # (cnt / N2) / (S1[si] / N1)
# B, 生成概率矩阵
for item in SW.items():
    si, wk = item[0]
    cnt = item[1]
    B[(si, wk)] = (cnt + 1) / (S1[si] + N1)   # (cnt / N_sw) / (S1[si] / N1)

print(N1, N2, N_sw, len(P), len(A), len(B))

PPS_list_valid = run(VALID_PATH)
print("PPS(valid):", np.mean(PPS_list_valid))