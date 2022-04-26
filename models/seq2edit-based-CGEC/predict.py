# -*- coding: utf-8
import os
from transformers import BertModel
import torch
import tokenization
import argparse
from gector.gec_model import GecBERTModel
import re
from opencc import OpenCC

cc = OpenCC("t2s")

def split_sentence(document: str, flag: str = "all", limit: int = 510):
    """
    Args:
        document:
        flag: Type:str, "all" 中英文标点分句，"zh" 中文标点分句，"en" 英文标点分句
        limit: 默认单句最大长度为510个字符
    Returns: Type:list
    """
    sent_list = []
    try:
        if flag == "zh":
            document = re.sub('(?P<quotation_mark>([。？！](?![”’"\'])))', r'\g<quotation_mark>\n', document)  # 单字符断句符
            document = re.sub('(?P<quotation_mark>([。？！])[”’"\'])', r'\g<quotation_mark>\n', document)  # 特殊引号
        elif flag == "en":
            document = re.sub('(?P<quotation_mark>([.?!](?![”’"\'])))', r'\g<quotation_mark>\n', document)  # 英文单字符断句符
            document = re.sub('(?P<quotation_mark>([?!.]["\']))', r'\g<quotation_mark>\n', document)  # 特殊引号
        else:
            document = re.sub('(?P<quotation_mark>([。？！….?!](?![”’"\'])))', r'\g<quotation_mark>\n', document)  # 单字符断句符
            document = re.sub('(?P<quotation_mark>(([。？！.!?]|…{1,2})[”’"\']))', r'\g<quotation_mark>\n',
                              document)  # 特殊引号

        sent_list_ori = document.splitlines()
        for sent in sent_list_ori:
            sent = sent.strip()
            if not sent:
                continue
            else:
                while len(sent) > limit:
                    temp = sent[0:limit]
                    sent_list.append(temp)
                    sent = sent[limit:]
                sent_list.append(sent)
    except:
        sent_list.clear()
        sent_list.append(document)
    return sent_list


def predict_for_file(input_file, output_file, model, batch_size, log=True, seg=False):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    sents = [s.strip() for s in lines]
    subsents = []
    s_map = []
    for i, sent in enumerate(sents):  # 将篇章划分为子句，分句预测再合并
        if seg:
            subsent_list = split_sentence(sent, flag="zh")
        else:
            subsent_list = [sent]
        s_map.extend([i for _ in range(len(subsent_list))])
        subsents.extend(subsent_list)
    assert len(subsents) == len(s_map)
    predictions = []
    cnt_corrections = 0
    batch = []
    for sent in subsents:
        batch.append(sent.split())
        if len(batch) == batch_size:  # 如果数据够了一个batch的话，
            preds, cnt = model.handle_batch(batch)
            assert len(preds) == len(batch)
            predictions.extend(preds)
            cnt_corrections += cnt
            if log:
                for z in zip(batch, preds):
                    print("source： " + "".join(z[0]))
                    print("target： " + "".join(z[1]))
                    print()
            batch = []

    if batch:
        preds, cnt = model.handle_batch(batch)
        assert len(preds) == len(batch)
        predictions.extend(preds)
        cnt_corrections += cnt
        if log:
            for z in zip(batch, preds):
                print("source： " + "".join(z[0]))
                print("target： " + "".join(z[1]))
                print()

    assert len(subsents) == len(predictions)
    if output_file:
        with open(output_file, 'w') as f1:
            with open(output_file + ".char", 'w') as f2:
                results = ["" for _ in range(len(sents))]
                for i, ret in enumerate(predictions):
                    ret_new = [tok.lstrip("##") for tok in ret]
                    ret = cc.convert("".join(ret_new))
                    results[s_map[i]] += cc.convert(ret)
                tokenizer = tokenization.FullTokenizer(vocab_file="vocab.txt", do_lower_case=False)
                for ret in results:
                    f1.write(ret + "\n")
                    line = tokenization.convert_to_unicode(ret)
                    tokens = tokenizer.tokenize(line)
                    f2.write(" ".join(tokens) + "\n")
    return cnt_corrections


def main(args):
    # get all paths
    model = GecBERTModel(vocab_path=args.vocab_path,
                         model_paths=args.model_path.split(','),
                         weights_names=args.weights_name.split(','),
                         max_len=args.max_len, min_len=args.min_len,
                         iterations=args.iteration_count,
                         min_error_probability=args.min_error_probability,
                         min_probability=args.min_error_probability,
                         log=False,
                         confidence=args.additional_confidence,
                         is_ensemble=args.is_ensemble,
                         weigths=args.weights,
                         cuda_device=args.cuda_device
                         )
    cnt_corrections = predict_for_file(args.input_file, args.output_file, model,
                                                   batch_size=args.batch_size, log=args.log, seg=args.seg)
    print(f"Produced overall corrections: {cnt_corrections}")


if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='Path to the model file',
                        required=True)  # GECToR模型文件，多个模型集成的话，可以用逗号隔开
    parser.add_argument('--weights_name',
                        help='Path to the pre-trained language model',
                        default='chinese-struct-bert',
                        required=True)  # 预训练语言模型文件，多个模型集成的话，每个模型对应一个PLM，可以用逗号隔开
    parser.add_argument('--vocab_path',
                        help='Path to the vocab file',
                        default='./data/output_vocabulary_chinese_char_hsk+lang8_5',
                        )  # 词表文件
    parser.add_argument('--input_file',
                        help='Path to the input file',
                        required=True)  # 输入文件，要求：预先分好词/字
    parser.add_argument('--output_file',
                        help='Path to the output file',
                        required=True)  # 输出结果文件
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=200)  # 最大输入长度（token数目），大于该长度的输入将被截断
    parser.add_argument('--min_len',
                        type=int,
                        help='The minimum sentence length'
                             '(all longer will be returned w/o changes)',
                        default=0)  # 最小修改长度（token数目），小于该长度的输入将不被修改
    parser.add_argument('--batch_size',
                        type=int,
                        help='The number of sentences in a batch when predicting',
                        default=128)  # 预测时的batch大小（句子数目）
    parser.add_argument('--iteration_count',
                        type=int,
                        help='The number of iterations of the model',
                        default=5)  # 迭代修改轮数
    parser.add_argument('--additional_confidence',
                        type=float,
                        help='How many probability to add to $KEEP token',
                        default=0.0)  # Keep标签额外置信度
    parser.add_argument('--min_probability',
                        type=float,
                        default=0)  # token级别最小修改阈值
    parser.add_argument('--min_error_probability',
                        type=float,
                        default=0.0)  # 句子级别最小修改阈值
    parser.add_argument('--is_ensemble', 
                        type=int,
                        help='Whether to do ensembling.',
                        default=0)  # 是否进行模型融合
    parser.add_argument('--weights',
                        help='Used to calculate weighted average', nargs='+',
                        default=None)  # 不同模型的权重（加权集成）
    parser.add_argument('--cuda_device',  
                        help='The number of GPU',
                        default=0)  # 使用GPU编号
    parser.add_argument('--log',  
                        action='store_true')  # 是否输出完整信息
    parser.add_argument('--seg',  
                        action='store_true')  # 是否切分长句预测后再合并
    args = parser.parse_args()
    main(args)
