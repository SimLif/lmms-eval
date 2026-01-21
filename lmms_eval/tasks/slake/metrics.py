from collections import defaultdict

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from .glossary import normalize_word


def split_sentence(sentence, n):
    words = defaultdict(int)
    # tmp_sentence = re.sub("[^a-zA-Z ]", "", sentence)
    tmp_sentence = sentence
    tmp_sentence = tmp_sentence.lower()
    tmp_sentence = tmp_sentence.strip().split()
    length = len(tmp_sentence)
    for i in range(length - n + 1):
        tmp_words = " ".join(tmp_sentence[i : i + n])
        if tmp_words:
            words[tmp_words] += 1
    return words


def calculate_exactmatch(candidate, reference):

    candidate = normalize_word(candidate)
    reference = normalize_word(reference)

    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    count = 0
    total = 0
    for word in reference_words:
        if word in candidate_words:
            count += 1
    for word in candidate_words:
        total += candidate_words[word]

    if total == 0:
        return 0  # "0 (warning: length of candidate's words is 0)"
    else:
        return count / total


def calculate_f1score(candidate, reference):
    candidate = normalize_word(candidate)
    reference = normalize_word(reference)

    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)

    word_set = set(candidate_words.keys()).union(reference_words.keys())

    tp = 0  # True Positive：重叠部分
    fp = 0  # False Positive：候选中超出的部分
    fn = 0  # False Negative：参考中缺失的部分

    for word in word_set:
        cand_count = candidate_words.get(word, 0)
        ref_count = reference_words.get(word, 0)
        common = min(cand_count, ref_count)  # 重叠部分按照最小计数统计
        tp += common
        fp += cand_count - common
        fn += ref_count - common

    if sum(candidate_words.values()) == 0 or sum(reference_words.values()) == 0 or tp == 0:
        return 0, 0, 0  # 返回 (F1, precision, recall)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def calculate_f1score_old(candidate, reference):

    candidate = normalize_word(candidate)
    reference = normalize_word(reference)

    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    word_set = set()
    for word in candidate_words:
        word_set.add(word)
    for word in reference_words:
        word_set.add(word)

    tp = 0
    fp = 0
    fn = 0
    for word in word_set:
        if word in candidate_words and word in reference_words:
            tp += candidate_words[word]
        elif word in candidate_words and word not in reference_words:
            fp += candidate_words[word]
        elif word not in candidate_words and word in reference_words:
            fn += reference_words[word]

    if len(candidate_words) == 0:
        return 0, 0, 0  # "0 (warning: length of candidate's words is 0)"
    elif len(reference_words) == 0:
        return 0, 0, 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if tp == 0:
            return 0, 0, 0
        else:
            return 2 * precision * recall / (precision + recall), precision, recall


def calculate_bleu(candidate, reference):
    # 归一化
    candidate = normalize_word(candidate)
    reference = normalize_word(reference)

    # 使用 nltk 的分词方式，也可以使用 split_sentence，但一般 BLEU 需要保留词序信息
    candidate_tokens = nltk.tokenize.word_tokenize(candidate)
    reference_tokens = nltk.tokenize.word_tokenize(reference)

    # 定义平滑函数，避免计算时因 n-gram 为零而返回 0
    smoothing = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing)
    return bleu_score
