from .bleu.bleu import Bleu
from .meteor import Meteor
from .cider.cider import Cider
from .rouge import Rouge

import jieba
import os
all_medical_nounts_file = os.path.join(os.path.dirname(__file__), "all_medical_nouns.txt")
jieba.load_userdict(all_medical_nounts_file)


def add_space_to_each_word_Jieba(report):
    # Using jieba to cut the Chinese word
    word_list = list(jieba.cut(report, cut_all=False, HMM=True))
    # Adding space between the Chinese word
    final_report = ""
    for w in word_list:
        final_report = final_report + w + " "
    return final_report


def add_space_to_each_word_llamaTokenizer(report, tokenizer):
    # 起始符号id
    start_token_id = int(tokenizer.encode("<|begin_of_text|>")[-1])  # 128000
    report_tokens = tokenizer.encode(report)
    final_report = ""
    for each_token in report_tokens:
        if int(each_token) == start_token_id:  # 如果是起始符号，则跳过，不参与报告组成 tokenizer.encode(stop_tokens)[-1]
            continue
        each_word = tokenizer.decode(each_token)
        final_report = final_report + each_word + " "
    return final_report


def add_space_to_each_word_single(report):
    final_report = ' '.join(report.replace(' ', '').replace('\n', ''))
    return final_report


def compute_scores(gts, res, method="English_word", tokenizer=None, temp=False):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    chosen_score = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"])
    ]

    if method == "jieba":
        # preprocess the sentences with jieba
        gts = {id:[add_space_to_each_word_Jieba(caption[0])] for id, caption in gts.items()}
        res = {id:[add_space_to_each_word_Jieba(caption[0])] for id, caption in res.items()}
    elif method == "llama":
        # preprocess the sentences with llama tokenizer
        gts = {id: [add_space_to_each_word_llamaTokenizer(caption[0], tokenizer)] for id, caption in gts.items()}
        res = {id: [add_space_to_each_word_llamaTokenizer(caption[0], tokenizer)] for id, caption in res.items()}
    elif method == "single_word":
        gts = {id: [add_space_to_each_word_single(caption[0])] for id, caption in gts.items()}
        res = {id: [add_space_to_each_word_single(caption[0])] for id, caption in res.items()}
    elif method == "English_word":
        gts = {id: [caption[0]] for id, caption in gts.items()}
        res = {id: [caption[0]] for id, caption in res.items()}
    else:
        print("Please select a way to tokenize coherent reports. You can choose 'jieba' or 'llama'.")

    if temp is False:
        eval_res = {}
        # Compute score for each metric
        for scorer, method in scorers:
            try:
                score, scores = scorer.compute_score(gts, res, verbose=0)
            except:
                score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, m in zip(score, method):
                    eval_res[m] = sc
            else:
                eval_res[method] = score
        return eval_res

    if temp is True:
        eval_res = {}
        # Compute score for each metric
        for scorer, method in chosen_score:
            try:
                score, scores = scorer.compute_score(gts, res, verbose=0)
            except TypeError:
                score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, m in zip(score, method):
                    eval_res[m] = sc
            else:
                eval_res[method] = score
        return eval_res
