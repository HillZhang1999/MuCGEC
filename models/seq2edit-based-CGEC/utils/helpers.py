import os
from collections import defaultdict, Counter
from pathlib import Path
import random
import string
from tqdm import tqdm
import json
from string import punctuation

chinese_punct = "……·——！―〉<>？｡。＂＃＄％＆＇（）＊＋，－／：《》；＜＝＞＠［’．＼］＾＿’｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"
english_punct = punctuation
letter = "123456789abcdefghijklmnopqrstuvwxyz"
FILTER = ["\x7f", " ", "\uf0e0", "\uf0a7", "\u200e", "\x8b", "\uf0b7", "\ue415", "\u2060", "\ue528", "\ue529", "ᩘ", "\ue074", "\x8b", "\u200c", "\ue529", "\ufeff", "\u200b", "\ue817", "\xad", '\u200f', '️', '่', '︎']
VOCAB_DIR = Path(__file__).resolve().parent.parent / "data"
PAD = "@@PADDING@@"
UNK = "@@UNKNOWN@@"
START_TOKEN = "$START"
SEQ_DELIMETERS = {"tokens": " ",
                  "labels": "SEPL|||SEPR",
                  "operations": "SEPL__SEPR",
                  "pos_tags": "SEPL---SEPR"}  # 分隔符，其中，如果一个source token被多次编辑，那么这些编辑label之间用"SEPL__SEPR"相分割
PUNCT = chinese_punct + english_punct + letter + letter.upper()

def split_char(line):
    """
    将中文按照字分开，英文按照词分开
    :param line: 输入句子
    :return: 分词后的句子
    """
    english = "abcdefghijklmnopqrstuvwxyz0123456789"
    output = []
    buffer = ""
    chinese_punct = "！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏."
    for s in line:
        if s in english or s in english.upper() or s in string.punctuation or s in chinese_punct:  # 英文或数字或标点不分
            buffer += s
        else:  # 中文或空格分
            if buffer and buffer != " ":
                output.append(buffer)
            buffer = ""
            if s != " ":
                output.append(s)
    if buffer:
        output.append(buffer)
    return output

def get_verb_form_dicts():
    """
    从词典verb-form-vocab.txt获得用于动词形式转换变形的encode和decode。
    verb-form-vocab.txt词典主要是存储了英文常见动词形式转换映射。
    :return:
    encode: key：单词形式转换， value:转换标签      likes_like:VBZ_VB
    decode: key:likes_VAZ_VB value:like      likes_VAZ_VB:like
    """
    path_to_dict = os.path.join(VOCAB_DIR, "verb-form-vocab.txt")
    encode, decode = {}, {}
    with open(path_to_dict, encoding="utf-8") as f:
        for line in f:
            words, tags = line.split(":")
            word1, word2 = words.split("_")
            tag1, tag2 = tags.split("_")
            decode_key = f"{word1}_{tag1}_{tag2.strip()}"
            if decode_key not in decode:
                encode[words] = tags
                decode[decode_key] = word2
    return encode, decode


ENCODE_VERB_DICT, DECODE_VERB_DICT = get_verb_form_dicts()  # 动词形式变换编码器、解码器


def get_target_sent_by_edits(source_tokens, edits):
    """
    对源句子token列表应用编辑操作（Span-level），得到目标句子token列表
    :param source_tokens: 源句子token列表
    :param edits: 编辑序列
    :return:目标句子token列表
    """
    target_tokens = source_tokens[:]
    shift_idx = 0
    for edit in edits:
        start, end, label, _ = edit
        target_pos = start + shift_idx
        source_token = target_tokens[target_pos] \
            if len(target_tokens) > target_pos >= 0 else ''
        if label == "":
            if target_tokens:
                del target_tokens[target_pos]
                shift_idx -= 1
        elif start == end:
            word = label.replace("$APPEND_", "")  # 添加操作
            target_tokens[target_pos: target_pos] = [word]
            shift_idx += 1
        elif label.startswith("$TRANSFORM_"):  # 变形操作
            word = apply_reverse_transformation(source_token, label)
            if word is None:
                word = source_token
            target_tokens[target_pos] = word
        elif start == end - 1:  # 替换操作
            word = label.replace("$REPLACE_", "")
            target_tokens[target_pos] = word
        elif label.startswith("$MERGE_"):  # 合并操作
            target_tokens[target_pos + 1: target_pos + 1] = [label]
            shift_idx += 1

    return replace_merge_transforms(target_tokens)  # 将Merge操作应用到目标句子token列表（当前只是用$Merge标签标记了需要合并的地方）


def replace_merge_transforms(tokens):
    """
    对序列应用Merge变形编辑（将当前token与下一个token合并）
    :param tokens: 词序列列表
    :return: Merge完成后的序列列表
    """
    if all(not x.startswith("$MERGE_") for x in tokens):
        return tokens
    target_tokens = tokens[:]
    allowed_range = range(1, len(tokens) - 1)
    for i in range(len(tokens)):
        target_token = tokens[i]
        if target_token.startswith("$MERGE"):
            if target_token.startswith("$MERGE_SWAP") and i in allowed_range:
                target_tokens[i - 1] = tokens[i + 1]
                target_tokens[i + 1] = tokens[i - 1]
    target_line = " ".join(target_tokens)
    target_line = target_line.replace(" $MERGE_HYPHEN ", "-")
    target_line = target_line.replace(" $MERGE_SPACE ", "")
    target_line = target_line.replace(" $MERGE_SWAP ", " ")
    return target_line.split()


def convert_using_case(token, smart_action):
    """
    对当前token进行大小写变换
    :param token: 当前token
    :param smart_action: 编辑操作标签
    :return: 编辑完成后的token
    """
    if not smart_action.startswith("$TRANSFORM_CASE_"):
        return token
    if smart_action.endswith("LOWER"):
        return token.lower()
    elif smart_action.endswith("UPPER"):
        return token.upper()
    elif smart_action.endswith("CAPITAL"):
        return token.capitalize()
    elif smart_action.endswith("CAPITAL_1"):
        return token[0] + token[1:].capitalize()
    elif smart_action.endswith("UPPER_-1"):
        return token[:-1].upper() + token[-1]
    else:
        return token


def convert_using_verb(token, smart_action):
    """
    对当前token进行动词时形式变换（人称、时态等）
    :param token: 当前token
    :param smart_action: 编辑操作标签
    :return: 编辑完成后的token
    """
    key_word = "$TRANSFORM_VERB_"
    if not smart_action.startswith(key_word):
        raise Exception(f"Unknown action type {smart_action}")
    encoding_part = f"{token}_{smart_action[len(key_word):]}"
    decoded_target_word = decode_verb_form(encoding_part)
    return decoded_target_word


def convert_using_split(token, smart_action):
    """
    对当前token进行切分（去掉连字符-）
    :param token: 当前token
    :param smart_action: 编辑操作标签
    :return: 编辑完成后的token
    """
    key_word = "$TRANSFORM_SPLIT"
    if not smart_action.startswith(key_word):
        raise Exception(f"Unknown action type {smart_action}")
    target_words = token.split("-")
    return " ".join(target_words)


# TODO 单复数变换不止有加s，还有加es的情况？
def convert_using_plural(token, smart_action):
    """
    对当前token进行单复数变换
    :param token: 当前token
    :param smart_action: 编辑操作标签
    :return: 编辑完成后的token
    """
    if smart_action.endswith("PLURAL"):
        return token + "s"
    elif smart_action.endswith("SINGULAR"):
        return token[:-1]
    else:
        raise Exception(f"Unknown action type {smart_action}")


def apply_reverse_transformation(source_token, transform):
    """
    对token进行转换操作
    :param source_token:
    :param transform:
    :return:
    """
    if transform.startswith("$TRANSFORM"):
        # deal with equal
        if transform == "$KEEP":  # 冗余？
            return source_token
        # deal with case
        elif transform.startswith("$TRANSFORM_CASE"):
            return convert_using_case(source_token, transform)
        # deal with verb
        elif transform.startswith("$TRANSFORM_VERB"):
            return convert_using_verb(source_token, transform)
        # deal with split
        elif transform.startswith("$TRANSFORM_SPLIT"):
            return convert_using_split(source_token, transform)
        # deal with single/plural
        elif transform.startswith("$TRANSFORM_AGREEMENT"):
            return convert_using_plural(source_token, transform)
        # raise exception if not find correct type
        raise Exception(f"Unknown action type {transform}")
    else:
        return source_token


def read_parallel_lines(fn1, fn2):
    """
    读取平行语料文件
    :param fn1: 源句子文件（纠错前）
    :param fn2: 目标句子文件（纠错后）
    :return: 分别包含源句子和目标句子的两个列表
    """
    lines1 = read_lines(fn1, skip_strip=True)
    lines2 = read_lines(fn2, skip_strip=True)
    assert len(lines1) == len(lines2), print(len(lines1), len(lines2))
    out_lines1, out_lines2 = [], []
    for line1, line2 in zip(lines1, lines2):
        if not line1.strip() or not line2.strip():
            continue
        else:
            out_lines1.append(line1)
            out_lines2.append(line2)
    return out_lines1, out_lines2


def read_lines(fn, skip_strip=False):
    """
    从文件中读取每一行
    :param fn: 文件路径
    :param skip_strip: 是否跳过空行
    :return: 包含文件中每一行的列表
    """
    if not os.path.exists(fn):
        return []
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [s.strip() for s in lines if s.strip() or skip_strip]

def write_lines(fn, lines, mode='w'):
    """
    将数据写入到文件中
    :param fn: 输出文件路径
    :param lines: 需要写入的数据
    :param mode: 写入的模式（w、a等）
    :return:
    """
    if mode == 'w' and os.path.exists(fn):
        os.remove(fn)
    with open(fn, encoding='utf-8', mode=mode) as f:
        f.writelines(['%s\n' % s for s in lines])


def decode_verb_form(original):
    return DECODE_VERB_DICT.get(original)


def encode_verb_form(original_word, corrected_word):
    decoding_request = original_word + "_" + corrected_word
    decoding_response = ENCODE_VERB_DICT.get(decoding_request, "").strip()
    if original_word and decoding_response:
        answer = decoding_response
    else:
        answer = None
    return answer