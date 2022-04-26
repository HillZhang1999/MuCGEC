import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BertTokenizer
from opencc import OpenCC
import re

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path')
parser.add_argument('-i', '--input_path')
parser.add_argument('-o', '--output_path')
parser.add_argument('-b', '--batch_size', default=50)
args = parser.parse_args()

cc = OpenCC("t2s")
tokenizer=BertTokenizer.from_pretrained(args.model_path)
model=BartForConditionalGeneration.from_pretrained(args.model_path)
model.eval()
model.half()
model.cuda()

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

def run_model(sents):
    num_ret_seqs = 1
    beam = 5
    inp_max_len = 100
    batch = [tokenizer(s, return_tensors='pt', padding='max_length', max_length=inp_max_len) for s in sents]
    oidx2bidx = {} #original index to final batch index
    final_batch = []
    for oidx, elm in enumerate(batch):
        if elm['input_ids'].size(1) <= inp_max_len:
            oidx2bidx[oidx] = len(final_batch)
            final_batch.append(elm)
    batch = {key: torch.cat([elm[key] for elm in final_batch], dim=0) for key in final_batch[0]}
    with torch.no_grad():
        generated_ids = model.generate(batch['input_ids'].cuda(),
                                attention_mask=batch['attention_mask'].cuda(),
                                num_beams=beam, num_return_sequences=num_ret_seqs, max_length=inp_max_len)
    _out = tokenizer.batch_decode(generated_ids.detach().cpu(), skip_special_tokens=True)
    outs = []
    for i in range(0, len(_out), num_ret_seqs):
        outs.append(_out[i:i+num_ret_seqs])
    final_outs = [[sents[oidx]] if oidx not in oidx2bidx else outs[oidx2bidx[oidx]] for oidx in range(len(sents))]
    return final_outs

def predict():
    sents = [l.strip() for l in open(args.input_path)]  # 分句
    subsents = []
    s_map = []
    for i, sent in enumerate(sents):  # 将篇章划分为子句，分句预测再合并
        subsent_list = split_sentence(sent, "zh")
        s_map.extend([i for _ in range(len(subsent_list))])
        subsents.extend(subsent_list)
    assert len(subsents) == len(s_map)
    b_size = args.batch_size
    outs = []
    for j in tqdm(range(0, len(subsents), b_size)):
        sents_batch = subsents[j:j+b_size]
        outs_batch = run_model(sents_batch)
        for sent, preds in zip(sents_batch, outs_batch):
            outs.append({'src': sent, 'preds': preds})
    results = ["" for _ in range(len(sents))]
    with open(args.output_path, 'w') as outf:
        for i, out in enumerate(outs):
            results[s_map[i]] += cc.convert(out['preds'][0])
        for res in results:
            outf.write(res + "\n")

predict()